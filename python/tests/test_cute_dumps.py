# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL kept files (CUTE_DSL_KEEP=ptx,cubin): one name per compilation, a collision is an
error (never an overwrite), and the startup hook covers processes that never import ptxv.
A stand-in `cutlass.base_dsl.compiler` names the kept files as CUTLASS DSL 4.6 does."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

PY = Path(__file__).resolve().parents[1]  # python/

FAKE = '''
import os

class _Keep:
    def __init__(self):
        self.value = True
        self.full_ptx_path = None
        self.full_cubin_path = None

class CompileOptions:
    def __init__(self):
        self.options = {"ptx": _Keep(), "cubin": _Keep()}

    def apply_envar_settings(self, envar, function_name):
        d = os.environ["CUTE_DSL_DUMP_DIR"]
        self.options["ptx"].full_ptx_path = os.path.join(d, f"{function_name}.sm_103a.ptx")
        self.options["cubin"].full_cubin_path = os.path.join(d, f"{function_name}.sm_103a.cubin")

def compile_and_keep(function_name):
    """One compilation: the kept PTX path it writes (and writes it)."""
    o = CompileOptions()
    o.apply_envar_settings(None, function_name)
    for k in o.options.values():
        for p in (k.full_ptx_path, k.full_cubin_path):
            if p:
                open(p, "w").write(function_name)
    return o.options["ptx"].full_ptx_path
'''


def fake_cutlass(tmp: Path, body: str = FAKE) -> Path:
    pkg = tmp / "fake" / "cutlass" / "base_dsl"
    pkg.mkdir(parents=True)
    (tmp / "fake" / "cutlass" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "compiler.py").write_text(body)
    (tmp / "dump").mkdir()
    return tmp / "fake"


def run(tmp: Path, code: str, pythonpath, keep="ptx,cubin"):
    env = dict(
        os.environ,
        CUTE_DSL_DUMP_DIR=str(tmp / "dump"),
        CUTE_DSL_KEEP=keep,
        PYTHONPATH=os.pathsep.join(str(p) for p in pythonpath),
        PTXV_NO_IMPORT_HOOK="1",
    )
    # The stand-in ahead of an installed CUTLASS DSL (whose .pth puts its packages first).
    pre = f"import sys; sys.path.insert(0, {str(tmp / 'fake')!r})\n"
    return subprocess.run(
        [sys.executable, "-c", pre + textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_two_kernels_of_one_name_keep_two_files(tmp_path):
    # Two compilations whose function names coincide (a module's main and tail kernels after
    # the DSL truncates their names) keep two PTX files, in this process and the next one.
    fake = fake_cutlass(tmp_path)
    code = """
        from ptxv import _ptxsrc
        _ptxsrc.install_cute_unique_dumps()
        from cutlass.base_dsl import compiler as c
        a = c.compile_and_keep("cutlass_FusedBackward_object_at")
        b = c.compile_and_keep("cutlass_FusedBackward_object_at")
        assert a != b, (a, b)
        print(a); print(b)
    """
    r1 = run(tmp_path, code, [PY, fake])
    r2 = run(tmp_path, code, [PY, fake])
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    ptx = sorted(p.name for p in (tmp_path / "dump").glob("*.ptx"))
    assert len(ptx) == 4, ptx
    assert all(p.startswith("cutlass_FusedBackward_object_at.p") for p in ptx), ptx


def test_a_collision_is_an_error_not_an_overwrite(tmp_path):
    fake = fake_cutlass(tmp_path)
    # An existing kept file under the name the next compilation gets (e.g. a dump directory
    # reused by an earlier process): DumpCollision, the file stays as it was.
    r = run(
        tmp_path,
        """
        import os
        from ptxv import _ptxsrc
        _ptxsrc.install_cute_unique_dumps()
        kept = os.path.join(os.environ["CUTE_DSL_DUMP_DIR"], f"k.{_ptxsrc.dump_token()}c0.sm_103a.ptx")
        open(kept, "w").write("first")
        from cutlass.base_dsl import compiler as c
        try:
            c.compile_and_keep("k")
        finally:
            assert open(kept).read() == "first"
    """,
        [PY, fake],
    )
    assert r.returncode != 0
    assert "DumpCollision" in r.stderr and "exists" in r.stderr, r.stderr


def test_a_cutedsl_without_the_hook_point_is_refused(tmp_path):
    fake = fake_cutlass(tmp_path, "class CompileOptions:\n    pass\n")
    r = run(
        tmp_path,
        """
        from ptxv import _ptxsrc
        _ptxsrc.install_cute_unique_dumps()
        import cutlass.base_dsl.compiler
    """,
        [PY, fake],
    )
    assert r.returncode != 0 and "apply_envar_settings" in r.stderr, r.stderr


def test_the_startup_hook_covers_processes_without_import_ptxv(tmp_path):
    # tools/ptxvcap-run (CUDA-injection capture): python/ptxv/_site first on PYTHONPATH.
    fake = fake_cutlass(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "sitecustomize.py").write_text(
        "import os\nos.environ['CHAINED'] = '1'\n"
    )
    r = run(
        tmp_path,
        """
        import os, sys
        from cutlass.base_dsl import compiler as c
        assert getattr(c.CompileOptions.apply_envar_settings, "_ptxv", False)
        assert "ptxv" not in sys.modules
        assert os.environ.get("CHAINED") == "1"
        a = c.compile_and_keep("k"); b = c.compile_and_keep("k")
        assert a != b
    """,
        [PY / "ptxv" / "_site", fake, tmp_path / "other"],
    )
    assert r.returncode == 0, r.stderr
    # Without CuTeDSL keeping PTX the hook does nothing.
    r = run(
        tmp_path,
        """
        from cutlass.base_dsl import compiler as c
        assert not getattr(c.CompileOptions.apply_envar_settings, "_ptxv", False)
    """,
        [PY / "ptxv" / "_site", fake],
        keep="",
    )
    assert r.returncode == 0, r.stderr
