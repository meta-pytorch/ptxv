# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates forms.ptx: one kernel per multimem form, compiled by ptxas to find the SASS each lowers to.
import sys

FORMS = []


def ldr(sem, scope, op, acc, vec, ty, nreg):
    FORMS.append(
        ("ldr", f"multimem.ld_reduce{sem}{scope}.global.{op}{acc}{vec}.{ty}", nreg)
    )


def st(sem, scope, vec, ty, nreg):
    FORMS.append(("st", f"multimem.st{sem}{scope}.global{vec}.{ty}", nreg))


def red(sem, scope, op, vec, ty, nreg):
    FORMS.append(("red", f"multimem.red{sem}{scope}.global.{op}{vec}.{ty}", nreg))


for sem in [
    "",
    ".weak",
    ".relaxed.sys",
    ".relaxed.gpu",
    ".acquire.sys",
    ".acquire.gpu",
    ".relaxed.cta",
    ".relaxed.cluster",
]:
    ldr(sem, "", "add", "", "", "f32", 1)
for op, acc, vec, ty, n in [
    ("add", "", ".v4", "f32", 4),
    ("add", "", ".v2", "f32", 2),
    ("add", "", "", "f16x2", 1),
    ("add", "", ".v4", "f16x2", 4),
    ("add", ".acc::f32", ".v4", "f16x2", 4),
    ("add", "", "", "bf16x2", 1),
    ("add", "", ".v4", "bf16x2", 4),
    ("add", ".acc::f32", ".v4", "bf16x2", 4),
    ("add", ".acc::f32", "", "bf16x2", 1),
    ("min", "", "", "bf16x2", 1),
    ("max", "", "", "bf16x2", 1),
    ("min", "", ".v4", "f16x2", 4),
    ("max", "", "", "f16x2", 1),
    ("min", "", "", "f32", 1),
    ("max", "", "", "f32", 1),
    ("add", "", "", "u32", 1),
    ("add", "", "", "s32", 1),
    ("min", "", "", "u32", 1),
    ("max", "", "", "s32", 1),
    ("add", "", "", "u64", 2),
    ("and", "", "", "b32", 1),
    ("or", "", "", "b32", 1),
    ("xor", "", "", "b32", 1),
    ("add", "", "", "f64", 2),
    ("add", ".acc::f16", ".v4", "e4m3x4", 4),
    ("add", ".acc::f16", ".v4", "e5m2x4", 4),
    ("add", "", "", "e4m3x4", 1),
    ("add", "", "", "f16", 1),
    ("add", "", "", "bf16", 1),
]:
    ldr(".relaxed", ".sys", op, acc, vec, ty, n)
for sem in [
    "",
    ".weak",
    ".relaxed.sys",
    ".release.sys",
    ".release.gpu",
    ".relaxed.gpu",
]:
    st(sem, "", ".v4", "f32", 4)
for vec, ty, n in [
    ("", "f32", 1),
    ("", "b32", 1),
    (".v2", "f32", 2),
    ("", "f16x2", 1),
    (".v4", "bf16x2", 4),
    ("", "u64", 2),
    ("", "f64", 2),
]:
    st(".relaxed", ".sys", vec, ty, n)
for sem in [
    "",
    ".relaxed.sys",
    ".release.sys",
    ".release.gpu",
    ".relaxed.gpu",
    ".relaxed.cta",
]:
    red(sem, "", "add", "", "s32", 1)
for op, vec, ty, n in [
    ("add", "", "u32", 1),
    ("add", "", "u64", 2),
    ("min", "", "u32", 1),
    ("max", "", "s32", 1),
    ("and", "", "b32", 1),
    ("or", "", "b32", 1),
    ("xor", "", "b32", 1),
    ("inc", "", "u32", 1),
    ("add", "", "f32", 1),
    ("add", ".v4", "f32", 4),
    ("add", "", "f16x2", 1),
    ("add", ".v4", "bf16x2", 4),
    ("min", "", "bf16x2", 1),
    ("max", "", "f16x2", 1),
    ("min", "", "f32", 1),
    ("add", "", "f64", 2),
    ("add", "", "bf16", 1),
]:
    red(".relaxed", ".sys", op, vec, ty, n)


def kernel(i, kind, ins, n):
    b64 = any(t in ins for t in (".u64", ".f64", ".s64"))
    rt = "b64" if b64 else "b32"
    k = 1 if b64 else n
    regs = ", ".join(f"%v{j}" for j in range(k))
    regs = "{" + regs + "}" if (".v" in ins) else regs
    if kind == "ldr":
        body = f"    {ins} {regs}, [%a];\n" + "".join(
            f"    st.global.{rt} [%o+{8 * j}], %v{j};\n" for j in range(k)
        )
    elif kind == "st":
        body = (
            "".join(f"    ld.global.{rt} %v{j}, [%o+{8 * j}];\n" for j in range(k))
            + f"    {ins} [%a], {regs};\n"
        )
    else:
        body = (
            "".join(f"    ld.global.{rt} %v{j}, [%o+{8 * j}];\n" for j in range(k))
            + f"    {ins} [%a], {regs};\n"
        )
    return f""".visible .entry f{i}(.param .u64 pa, .param .u64 po) {{
    .reg .{rt} %v<4>; .reg .b64 %a, %o;
    ld.param.u64 %a, [pa]; ld.param.u64 %o, [po];
{body}    ret;
}}
"""


if __name__ == "__main__":
    which = int(sys.argv[1]) if len(sys.argv) > 1 else None
    hdr = ".version 9.0\n.target sm_103a\n.address_size 64\n"
    if which is None:
        for i, (k, ins, n) in enumerate(FORMS):
            print(i, ins)
    else:
        k, ins, n = FORMS[which]
        print(hdr + kernel(which, k, ins, n))
