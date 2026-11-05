# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`python -m ptxv`:

python -m ptxv run script.py [args...]   run a script with ptxv imported first, so the
                                         CUPTI module hook sees every module it loads
python -m ptxv daemon [options]          ptxvd (python -m ptxv.daemon --help)
python -m ptxv check CASE                verdict of a case directory
python -m ptxv compare REF CAND          equivalence of two case directories
"""

from __future__ import annotations

import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        if not rest:
            print("usage: python -m ptxv run script.py [args...]", file=sys.stderr)
            return 2
        import ptxv as _ptxv  # noqa: F401  (installs the module hook before the script's CUDA work)

        sys.argv = rest
        runpy.run_path(rest[0], run_name="__main__")
        return 0
    if cmd == "daemon":
        from .daemon import main as dmain

        dmain(rest)
        return 0
    if cmd == "check" and len(rest) == 1:
        import ptxv

        print(ptxv.check(rest[0]))
        return 0
    if cmd == "compare" and len(rest) == 2:
        import ptxv

        eq = ptxv.compare(rest[0], rest[1])
        print(eq.headline)
        print(eq)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
