/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Every source mutant of tools/mutate/srcmut.py finds its text exactly once in this tree, and
//! not only inside a comment (`srcmut.py --check-anchors`, which builds and runs nothing). A
//! change to code a mutant anchors on re-points the mutant's find text in the same commit, or
//! this test fails and names the mutant. A mutant whose find text is gone can no longer be
//! applied, and srcmut reports it as an error only when the mutant is run. The mutation tools'
//! own tests (`pytest tools/mutate`) run here too: they check that --check-anchors finds a
//! broken anchor on the path the check takes (no --root, several mutants, the whole list).

use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

/// The positive control's mutant (an entry of srcmut.py's own) and its file.
const CONTROL: &str = "srcmut-none-chosen-ok";
const CONTROL_FILE: &str = "tools/mutate/srcmut.py";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `$PTXV_PYTHON -I -B -m pytest -q ... ARGS tools/mutate` (default `python3`) in the repository
/// root, and pytest's closing line (`summary`). Neither the environment nor a parent directory
/// changes what runs: -I (no PYTHONPATH, user site or other PYTHON* variable, no current
/// directory on sys.path), PYTEST_ADDOPTS and PYTEST_PLUGINS removed, plugin autoload off, the
/// in-tree tools/mutate/pytest.ini pinned (-c: no parent pytest.ini, tox.ini, setup.cfg or
/// pyproject.toml) and conftest.py files read only from tools/mutate down (--confcutdir). -B and
/// no cache provider write nothing into the tree.
fn pytest(args: &[&str]) -> (Output, String) {
    let python = std::env::var("PTXV_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .args([
            "-I",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--color=no",
        ])
        .args([
            "-c",
            "tools/mutate/pytest.ini",
            "--confcutdir",
            "tools/mutate",
        ])
        .args(args)
        .arg("tools/mutate")
        .env_remove("PYTEST_ADDOPTS")
        .env_remove("PYTEST_PLUGINS")
        .env("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        .current_dir(repo())
        .output()
        .unwrap_or_else(|e| panic!("{python} -m pytest: {e}"));
    let line = summary(&out);
    (out, line)
}

/// pytest's closing line (`-q`) without its duration and warning count: "13 passed",
/// "1 passed, 12 deselected" (a warning does not change what ran). "" without one.
fn summary(out: &Output) -> String {
    let text = String::from_utf8_lossy(&out.stdout);
    let last = text
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .unwrap_or("");
    let body = last.rfind(" in ").map_or(last, |i| &last[..i]);
    body.split(", ")
        .filter(|p| !p.ends_with(" warning") && !p.ends_with(" warnings"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `$PTXV_PYTHON -I tools/mutate/srcmut.py ARGS` (default `python3`) in the repository root.
fn srcmut(args: &[&str]) -> Output {
    let root = repo();
    let srcmut = root.join("tools/mutate/srcmut.py");
    let python = std::env::var("PTXV_PYTHON").unwrap_or_else(|_| "python3".to_string());
    Command::new(&python)
        .arg("-I")
        .arg(&srcmut)
        .args(args)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("{python} {}: {e}", srcmut.display()))
}

/// `srcmut.py --check-anchors` exits 0. Pass or fail is its exit status; its output only names
/// the broken mutants. Two controls come first. Negative: the check of a mutant that does not
/// exist, and the check with an --only that names nothing, must exit 2 (srcmut's --only error),
/// so that an interpreter that exits the same whatever it is given (PTXV_PYTHON=/bin/true, say)
/// or an --only path that checks anchors instead cannot pass. Any error is not enough: in a
/// tree with a mutant of srcmut.py applied, that mutant's own anchor is gone, so a check of
/// every mutant exits 1. Positive: the check of one mutant (CONTROL, a .py file) against a
/// scratch tree that holds its file (`--root`) exits 0 with the file as it is and 1 with a
/// broken anchor planted (the file twice, so that the find text occurs twice; the file emptied;
/// no file), so that a check that finds no broken anchor at all, or that reads another tree than
/// --root names, cannot pass. That control takes one mutant through --root; the check below
/// takes every mutant without --root. A check that skips mutants (all but the first, those of
/// .rs files, a tail of the list, all but one per file), that checks nothing without --root or
/// that takes an anchor kept in a comment passes both, and fails the_mutation_tools_tests_pass
/// instead.
#[test]
fn every_srcmut_mutant_finds_its_text_once() {
    for only in ["no-such-mutant", ""] {
        let none = srcmut(&["--check-anchors", "--only", only]);
        assert_eq!(
            none.status.code(),
            Some(2),
            "srcmut.py --check-anchors --only {only:?} exited with {}, not with the --only \
             error (2), so the check below could pass without checking anything:\n{}{}",
            none.status,
            String::from_utf8_lossy(&none.stdout),
            String::from_utf8_lossy(&none.stderr)
        );
    }
    let text = std::fs::read_to_string(repo().join(CONTROL_FILE)).unwrap();
    let scratch = std::env::temp_dir().join(format!("ptxv-srcmut-anchors-{}", std::process::id()));
    let file = scratch.join(CONTROL_FILE);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    for (planted, want, what) in [
        (Some(text.clone()), 0, "the file as it is"),
        (Some(text.repeat(2)), 1, "the file twice"),
        (Some(String::new()), 1, "the file emptied"),
        (None, 1, "no file"),
    ] {
        match planted {
            Some(t) => std::fs::write(&file, t).unwrap(),
            None => std::fs::remove_file(&file).unwrap(),
        }
        let root = scratch.to_str().unwrap();
        let out = srcmut(&["--check-anchors", "--only", CONTROL, "--root", root]);
        let why = if want == 0 {
            format!("{CONTROL} must stay a mutant of {CONTROL_FILE}")
        } else {
            "the check passed a broken anchor".to_string()
        };
        assert_eq!(
            out.status.code(),
            Some(want),
            "positive control, {what}: srcmut.py --check-anchors --only {CONTROL} --root {root} \
             exited with {}, not {want} ({why}):\n{}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
    let out = srcmut(&["--check-anchors"]);
    assert!(
        out.status.success(),
        "srcmut.py --check-anchors exited with {}; re-point each mutant below in the commit \
         that changed its code:\n{}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The tests of tools/mutate/test_mutate.py that this test's coverage claims rest on.
/// test_srcmut_check_anchors runs --check-anchors without --root over two mutants of .rs files,
/// the broken one second (a check that skips mutants, all but the first or those of .rs files,
/// or that checks nothing without --root fails it), and over anchors kept only inside a
/// comment. test_srcmut_anchors_whole_list runs it over the whole list in a scratch tree with
/// the last mutant's anchor and a later mutant's anchor in a shared file broken (a check that
/// drops a tail of the list, or checks one mutant per file, fails it).
const KEY_TESTS: [&str; 2] = [
    "test_srcmut_check_anchors",
    "test_srcmut_anchors_whole_list",
];

/// The mutation tools' own tests pass (`pytest tools/mutate`, see `pytest`; PTXV_PYTHON must have
/// pytest): pytest's closing line must read "N passed" and nothing else, so a skipped, deselected
/// or failing test fails here; the check in every_srcmut_mutant_finds_its_text_once passes a
/// check that skips mutants with an anchor gone. Then each of KEY_TESTS, selected by name
/// (`-k`), must report "1 passed, N - 1 deselected": it exists, runs and passes. A negative
/// control last: a selection that matches no test must exit 5 (no test ran) with "N
/// deselected", so that an interpreter that exits the same whatever it is given, or one without
/// pytest, cannot pass.
#[test]
fn the_mutation_tools_tests_pass() {
    let show = |o: &Output| {
        format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        )
    };
    let (all, line) = pytest(&[]);
    let n: usize = line
        .strip_suffix(" passed")
        .and_then(|x| x.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            panic!(
                "pytest tools/mutate: closing line {line:?}, not \"N passed\" (PTXV_PYTHON must \
                 have pytest; no test may fail, skip or be deselected), exit {}:\n{}",
                all.status,
                show(&all)
            )
        });
    assert!(
        all.status.success(),
        "pytest tools/mutate exited with {}:\n{}",
        all.status,
        show(&all)
    );
    for key in KEY_TESTS {
        let (out, line) = pytest(&["-k", key]);
        let want = format!("1 passed, {} deselected", n - 1);
        assert!(
            out.status.success() && line == want,
            "pytest tools/mutate -k {key}: closing line {line:?}, not {want:?} (the test must \
             exist, run and pass), exit {}:\n{}",
            out.status,
            show(&out)
        );
    }
    let (none, line) = pytest(&["-k", "no_such_test"]);
    let want = format!("{n} deselected");
    assert!(
        none.status.code() == Some(5) && line == want,
        "pytest tools/mutate -k no_such_test: exit {}, closing line {line:?}, not 5 and {want:?} \
         (no test ran), so the runs above could pass without testing anything:\n{}",
        none.status,
        show(&none)
    );
}
