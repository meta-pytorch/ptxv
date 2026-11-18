#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export real Machine memory effects from an instrumented source archive."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
from pathlib import Path

LIMIT = 48 * 1024**3
SOURCE_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/sim/src/exec.rs",
    "crates/sim/src/exec/atomic_components.rs",
    "crates/sim/src/atomic_component_tests.rs",
    "crates/sim/src/dpor.rs",
    "crates/sim/src/wellsync.rs",
    "crates/sim/src/wellsync/ticket.rs",
    "crates/sim/src/coverage.rs",
    "crates/sim/src/num.rs",
    "crates/sim/src/halfin.rs",
    "crates/sim/src/machine.rs",
    "crates/sim/src/explore.rs",
    "crates/ptx/src/lib.rs",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    resource.setrlimit(resource.RLIMIT_DATA, (LIMIT, LIMIT))


def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="archived ptxv source"
    )
    parser.add_argument(
        "--work", type=Path, required=True, help="new temporary directory"
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--cargo-lock", type=Path, help="cached offline lockfile override"
    )
    parser.add_argument("--allow-known-wide-atomic-bug", action="store_true")
    lean = parser.add_mutually_exclusive_group()
    lean.add_argument("--lean-replay", type=Path, help="built Lean replay executable")
    lean.add_argument(
        "--lean-project", type=Path, help="Lake project containing Replay.lean"
    )
    config = parser.parse_args()
    for name in (
        "source",
        "work",
        "target_dir",
        "cargo_lock",
        "lean_replay",
        "lean_project",
    ):
        path = getattr(config, name)
        if path is not None:
            setattr(config, name, path.resolve())
    return config


def prepare_archive(
    config: argparse.Namespace, harness: Path
) -> tuple[Path, dict[str, object]]:
    source = config.source
    work = config.work
    if work.exists():
        raise ValueError(f"work directory already exists: {work}")
    work.mkdir(parents=True)
    archived = work / "source"
    shutil.copytree(source, archived)
    receipt: dict[str, object] = {
        "revision": config.revision,
        "harness_sha256": digest(harness),
        "source_sha256": {name: digest(source / name) for name in SOURCE_FILES},
    }
    if config.cargo_lock:
        receipt["cargo_lock_override"] = {
            "path": str(config.cargo_lock),
            "source_sha256": digest(source / "Cargo.lock"),
            "override_sha256": digest(config.cargo_lock),
        }
        shutil.copyfile(config.cargo_lock, archived / "Cargo.lock")
    return archived, receipt


def instrument_archive(archived: Path, harness: Path) -> dict[str, dict[str, str]]:
    patches = {
        "num.rs": (
            "\n#[cfg(test)]\n#[path = "
            + json.dumps(str(harness))
            + "]\nmod production_replay;\n"
        ),
        "halfin.rs": """
#[cfg(test)]
impl HalfState {
    pub(crate) fn production_replay_raw(&self, key: (u8, u64, u64)) -> (Option<Src>, bool) {
        (self.mem.get(&key).copied(), self.gwritten.contains(&(key.1, key.2)))
    }
}
""",
    }
    recorded: dict[str, dict[str, str]] = {}
    for filename, patch in patches.items():
        path = archived / "crates" / "sim" / "src" / filename
        original = digest(path)
        with path.open("a") as output:
            output.write(patch)
        recorded[filename] = {
            "source_sha256": original,
            "patch": patch,
            "patched_sha256": digest(path),
        }
    return recorded


def export_production(
    config: argparse.Namespace, archived: Path, receipt: dict[str, object]
) -> Path:
    work = config.work
    corpus = work / "corpus.json"
    env = os.environ | {
        "CARGO_TARGET_DIR": str(config.target_dir),
        "CUDA_VISIBLE_DEVICES": "",
        "PTXV_EXCLUDE_GPUS": "0,1,2,3,4,5,6,7",
        "PTXV_PRODUCTION_REPLAY_OUTPUT": str(corpus),
        "PTXV_ALLOW_KNOWN_WIDE_ATOMIC_BUG": (
            "1" if config.allow_known_wide_atomic_bug else "0"
        ),
    }
    command = [
        "cargo",
        "test",
        "--offline",
        "--manifest-path",
        str(archived / "Cargo.toml"),
        "-p",
        "ptxv-sim",
        "--lib",
        "production_replay::export_production_replay",
        "--",
        "--nocapture",
    ]
    receipt["command"] = command
    (work / "instrumentation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (work / "production.log").open("w") as log:
        subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=limit_memory,
            check=True,
        )
    receipt["corpus_sha256"] = digest(corpus)
    receipt["cases"] = len(json.loads(corpus.read_text())["cases"])
    return corpus


def replay_lean(config: argparse.Namespace, corpus: Path) -> dict[str, str]:
    if config.lean_replay:
        command = [str(config.lean_replay), str(corpus)]
        project = None
        replay_source = config.lean_replay
    elif config.lean_project:
        command = ["lake", "env", "lean", "--run", "Replay.lean", str(corpus)]
        project = config.lean_project
        replay_source = project / "Replay.lean"
    else:
        return {}
    with (config.work / "lean-replay.log").open("w") as log:
        subprocess.run(
            command,
            cwd=project,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=limit_memory,
            check=True,
        )
    return {"lean_replay_sha256": digest(replay_source)}


def main() -> None:
    config = parse_config()
    harness = Path(__file__).with_name("production_replay.rs").resolve()
    archived, receipt = prepare_archive(config, harness)
    receipt["patches"] = instrument_archive(archived, harness)
    corpus = export_production(config, archived, receipt)
    receipt.update(replay_lean(config, corpus))
    (config.work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": receipt["cases"], "corpus": str(corpus)}))


if __name__ == "__main__":
    main()
