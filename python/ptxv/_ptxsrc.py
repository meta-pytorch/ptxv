"""PTX source of captured kernels.

A captured module image is SASS only (CUPTI hands over the cubin the driver loaded). The PTX it
was compiled from survives when the toolchain kept it:

- Triton: `<cache>/<hash>/<name>.ptx` next to `<name>.cubin` (TRITON_CACHE_DIR, else
  ~/.triton/cache);
- CuTeDSL: `CUTE_DSL_KEEP=ptx,cubin` (or the deprecated CUTE_DSL_KEEP_PTX=1 with KEEP_CUBIN)
  writes `<function>.<arch>.ptx` and `.cubin` to CUTE_DSL_DUMP_DIR (default: the cwd);
- anything else (nvcc `-keep`, `nvcc -ptx` + ptxas): directories in PTXV_PTX_DIRS
  (colon-separated) holding `<x>.ptx` next to a cubin compiled from it.

A PTX file belongs to a captured kernel only if a cubin compiled from it holds the same machine
code for that kernel: the SHA-1 of the `.text.<kernel>` section must be equal (pruned case
cubins keep their own kernel's section intact). A name match alone is never used.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import struct
from pathlib import Path


def _sections(d: bytes):
    shoff = struct.unpack_from("<Q", d, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]
    stro = secs[shstrndx][4]

    def cstr(off):
        return bytes(d[off : d.index(b"\0", off)]).decode(errors="replace")

    return {cstr(stro + s[0]): (s[4], s[5], s[1]) for s in secs}


def text_keys(data: bytes) -> dict[str, str]:
    """Function name -> SHA-1 of its `.text.<name>` section, for an ELF cubin."""
    if data[:4] != b"\x7fELF":
        return {}
    try:
        secs = _sections(data)
    except (struct.error, ValueError, IndexError):
        return {}
    return {
        n[6:]: hashlib.sha1(data[off : off + size]).hexdigest()
        for n, (off, size, typ) in secs.items()
        if n.startswith(".text.") and typ != 8 and size  # not SHT_NOBITS
    }


def ptx_entries(src: str) -> list[str]:
    return re.findall(r"\.entry\s+([A-Za-z0-9_$.]+)", src)


def _read(p: str) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def _cute_dump_dirs() -> list[str]:
    keep = os.environ.get("CUTE_DSL_KEEP", "")
    if "ptx" not in keep.split(",") and os.environ.get("CUTE_DSL_KEEP_PTX", "0") in (
        "0",
        "",
    ):
        return []
    return [os.environ.get("CUTE_DSL_DUMP_DIR") or os.getcwd()]


def _pair_dirs() -> list[str]:
    return _cute_dump_dirs() + [
        d for d in os.environ.get("PTXV_PTX_DIRS", "").split(":") if d
    ]


def find_ptx(cubins: dict[str, bytes]) -> dict[str, str]:
    """Kernel name -> PTX text for every captured kernel (name -> its module image) whose PTX
    one of the sources above kept."""
    want = {}
    for n, img in cubins.items():
        k = text_keys(img).get(n)
        if k:
            want[n] = k
    out: dict[str, str] = {}
    # Triton: the cache entry is found by kernel name.
    root = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    if os.path.isdir(root):
        for n, k in want.items():
            for c in glob.glob(os.path.join(root, "*", f"{n}.cubin")):
                p = c[: -len(".cubin")] + ".ptx"
                try:
                    if os.path.exists(p) and text_keys(_read(c)).get(n) == k:
                        out[n] = Path(p).read_text()
                        break
                except OSError:
                    continue
    # CuTeDSL dumps / other kept pairs: any cubin in the directory, its PTX by stem.
    left = {n: k for n, k in want.items() if n not in out}
    for d in _pair_dirs() if left else []:
        for c in glob.glob(os.path.join(d, "**", "*.cubin"), recursive=True):
            try:
                keys = text_keys(_read(c))
            except OSError:
                continue
            for n in [n for n, k in left.items() if keys.get(n) == k]:
                stem = c[: -len(".cubin")]
                for p in (stem + ".ptx", *glob.glob(stem.rsplit(".", 1)[0] + "*.ptx")):
                    if os.path.exists(p):
                        src = Path(p).read_text(errors="replace").rstrip("\x00")
                        if n in ptx_entries(src):
                            out[n] = src
                            left.pop(n)
                            break
            if not left:
                break
    return out


_HEADER = re.compile(r"^[ \t]*\.(version|target|address_size)\b[^\n]*$", re.M)
_DECL = re.compile(
    r"^(\.extern|\.global|\.shared|\.const|\.visible\s+\.global|\.weak\s+\.global)\b"
)


def _strip_debug(src: str) -> str:
    src = re.sub(r"^[ \t]*\.(file|loc)\b[^\n]*$", "", src, flags=re.M)
    out, i = [], 0
    for m in re.finditer(r"^[ \t]*\.section\s+\.debug\w*\s*\{", src, re.M):
        if m.start() < i:
            continue
        depth, k = 0, m.end() - 1
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(src[i : m.start()])
        i = k + 1
    out.append(src[i:])
    return "".join(out)


def merge_ptx(srcs: list[str]) -> str:
    """One PTX module holding every kernel of several (a multi-kernel case is one PTX text).
    Header of the module with the highest .version; module-scope declarations and `.func`
    definitions that repeat verbatim are kept once; the same name declared differently raises
    ValueError."""
    srcs = list(dict.fromkeys(srcs))
    if len(srcs) == 1:
        return srcs[0]
    # Debug info does not merge (`.file` indices and `.section .debug_*` repeat per module):
    # drop it; it carries no semantics.
    srcs = [_strip_debug(x) for x in srcs]

    def ver(s):
        m = re.search(r"^\s*\.version (\d+)\.(\d+)", s, re.M)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    first = max(srcs, key=ver)
    head = "\n".join(m.group(0) for m in _HEADER.finditer(first))
    decls: dict[str, str] = {}
    funcs: dict[str, str] = {}
    body = []
    # Module-scope variables declared differently in two modules (a function-static of two
    # macro variants of one source) are module-local: rename them in the later module.
    seen_text: dict[str, str] = {}
    for mi, s in enumerate(srcs):
        own = {}
        for line in s.split("\n"):
            t = line.strip()
            if _DECL.match(t) and t.endswith(";") and not t.startswith(".extern"):
                nm = re.findall(
                    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\[[^\]]*\]\s*)?(?:;|=)", t
                )
                if nm:
                    own[nm[0]] = t
        for nm, t in own.items():
            if nm in seen_text and seen_text[nm] != t:
                s = re.sub(
                    r"(?<![A-Za-z0-9_$])" + re.escape(nm) + r"(?![A-Za-z0-9_$])",
                    f"{nm}__m{mi}",
                    s,
                )
            else:
                seen_text.setdefault(nm, t)
        srcs[mi] = s
    for s in srcs:
        s = _HEADER.sub("", s)
        lines = s.split("\n")
        i = 0
        out = []
        while i < len(lines):
            t = lines[i].strip()
            if _DECL.match(t) and t.endswith(";"):
                nm = re.findall(
                    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\[[^\]]*\]\s*)?(?:;|=)", t
                )
                nm = nm[0] if nm else t
                if nm in decls:
                    if decls[nm] != t:
                        raise ValueError(
                            f"PTX merge: module-scope {nm} declared differently"
                        )
                    i += 1
                    continue
                decls[nm] = t
                out.append(lines[i])
                i += 1
                continue
            m = re.match(
                r"^(\.visible\s+|\.weak\s+)?\.func\b.*?([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
                t,
            )
            if m and not t.endswith(";"):
                # A .func definition: through its matching closing brace.
                j, depth, seen = i, 0, False
                while j < len(lines):
                    depth += lines[j].count("{") - lines[j].count("}")
                    seen = seen or "{" in lines[j]
                    j += 1
                    if seen and depth == 0:
                        break
                text = "\n".join(lines[i:j])
                nm = m.group(2)
                if nm in funcs:
                    if funcs[nm] != text:
                        raise ValueError(f"PTX merge: .func {nm} defined differently")
                else:
                    funcs[nm] = text
                    out.append(text)
                i = j
                continue
            if m and t.endswith(";"):  # prototype
                if "proto:" + m.group(2) in decls:
                    i += 1
                    continue
                decls["proto:" + m.group(2)] = t
            out.append(lines[i])
            i += 1
        body.append("\n".join(out))
    return head + "\n" + "\n".join(body)


def install_nvcc_hook(root: str) -> None:
    """JIT builders that run `nvcc -cubin ...` from Python (mscclpp's KernelBuilder for DeepEP)
    keep no PTX. With this hook every such successful call is repeated with `-ptx` (same
    flags, macros and virtual architecture) and the pair `<root>/<hash>/<name>.{ptx,cubin}` is
    kept for find_ptx (PTXV_PTX_DIRS). Enabled by PTXV_NVCC_PTX_DIR (ptxv/__init__)."""
    import shutil
    import subprocess

    if getattr(subprocess.run, "_ptxv_nvcc", False):
        return
    orig = subprocess.run

    def run(*args, **kw):
        res = orig(*args, **kw)
        try:
            argv = args[0] if args else kw.get("args")
            if (
                isinstance(argv, (list, tuple))
                and argv
                and os.path.basename(str(argv[0])) == "nvcc"
                and "-cubin" in argv
                and "-o" in argv
                and getattr(res, "returncode", 1) == 0
            ):
                argv = [str(a) for a in argv]
                out = argv[argv.index("-o") + 1]
                h = hashlib.sha1("\0".join(argv).encode()).hexdigest()[:16]
                d = os.path.join(root, h)
                os.makedirs(d, exist_ok=True)
                stem = os.path.join(
                    d,
                    os.path.basename(out)[: -len(".cubin")]
                    if out.endswith(".cubin")
                    else os.path.basename(out),
                )
                shutil.copyfile(out, stem + ".cubin")
                pargv = [a for a in argv if not a.startswith("--gpu-code=")]
                pargv[pargv.index("-cubin")] = "-ptx"
                pargv[pargv.index("-o") + 1] = stem + ".ptx"
                orig(pargv, capture_output=True, stdin=subprocess.DEVNULL, check=False)
        except Exception:  # noqa: BLE001 (best effort: never break the program)
            pass
        return res

    run._ptxv_nvcc = True  # type: ignore[attr-defined]
    subprocess.run = run
    os.environ.setdefault("PTXV_PTX_DIRS", root)
    if root not in os.environ["PTXV_PTX_DIRS"].split(":"):
        os.environ["PTXV_PTX_DIRS"] += ":" + root


def install_cute_unique_dumps() -> None:
    """CuTeDSL names its kept files `<function name>.<arch>.{ptx,cubin}` after a truncated
    function name, so specializations sharing that prefix overwrite each other's PTX and cubin
    (CUTLASS DSL 4.6). Give every compilation its own name (`<name>.p<pid>c<n>`) as soon as
    `cutlass.base_dsl.compiler` is imported. Only the dump paths change."""
    import importlib.abc
    import itertools
    import sys

    count = itertools.count()

    def patch(mod):
        cls = getattr(mod, "CompileOptions", None)
        f = getattr(cls, "apply_envar_settings", None)
        if f is None or getattr(f, "_ptxv", False):
            return

        def apply_envar_settings(self, envar, function_name, *a, **kw):
            return f(
                self, envar, f"{function_name}.p{os.getpid()}c{next(count)}", *a, **kw
            )

        apply_envar_settings._ptxv = True  # type: ignore[attr-defined]
        cls.apply_envar_settings = apply_envar_settings

    name = "cutlass.base_dsl.compiler"
    if name in sys.modules:
        patch(sys.modules[name])
        return

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            try:
                import importlib.util

                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def run(module):
                exec_module(module)
                patch(module)
                if self in sys.meta_path:
                    sys.meta_path.remove(self)

            spec.loader.exec_module = run  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, Finder())


def keep_entries(src: str, keep: set[str]) -> str:
    """The module without the `.entry` definitions not in `keep` (several modules of one
    source compiled with different macros each contribute only the kernels launched from them)."""
    out, i = [], 0
    for m in re.finditer(
        r"^[ \t]*(?:\.visible\s+|\.weak\s+)?\.entry\s+([A-Za-z0-9_$.]+)", src, re.M
    ):
        if m.start() < i:
            continue
        if m.group(1) in keep:
            continue
        j = src.index("{", m.end())
        depth = 0
        for k in range(j, len(src)):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    break
        out.append(src[i : m.start()])
        i = k + 1
    out.append(src[i:])
    return "".join(out)
