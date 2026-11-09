/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The hardware targets under `targets/` and `fb/targets/` (layout and rules:
//! targets/README.md): each directory's `target.toml`, its assumption registry and the
//! consistency checks that let targets be written independently.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;

/// Target statuses, in order of maturity (targets/README.md).
pub const STATUSES: &[&str] = &["scaffold", "spec", "lifting", "checking", "verified"];
/// Assumption statuses (top-level ASSUMPTIONS.md).
pub const ASSUMPTION_STATUSES: &[&str] =
    &["enforced", "validated", "argued", "refuted", "unvalidated"];
/// Assumption kinds: `spec` (written before code relies on it), `code`, `process`.
pub const KINDS: &[&str] = &["spec", "code", "process"];
/// Assumption classes (the section letters of the top-level ASSUMPTIONS.md).
pub const CLASSES: &[&str] = &["E", "S", "L", "N", "H"];
/// Directories searched for targets, relative to the repository root.
pub const DIRS: &[&str] = &["targets", "fb/targets"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpec {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub isa: Vec<String>,
    pub arch: Vec<String>,
    pub id_prefix: String,
    pub status: String,
    #[serde(default)]
    pub based_on: Option<String>,
    /// Assumption registry, relative to the target directory (default `ASSUMPTIONS.md`).
    #[serde(default)]
    pub assumptions: Option<String>,
    /// Model, relative to the target directory (default `SPEC.md`).
    #[serde(default)]
    pub spec: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Target {
    pub dir: PathBuf,
    pub spec: TargetSpec,
}

impl Target {
    pub fn assumptions_path(&self) -> PathBuf {
        self.dir
            .join(self.spec.assumptions.as_deref().unwrap_or("ASSUMPTIONS.md"))
    }
    pub fn spec_path(&self) -> PathBuf {
        self.dir
            .join(self.spec.spec.as_deref().unwrap_or("SPEC.md"))
    }
    /// The template directory, which is validated but owns no IDs.
    pub fn is_template(&self) -> bool {
        self.spec.id == "_template"
    }
}

/// One `### ID · kind` entry of an assumption registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub kind: String,
    /// The value of its `- Status: **...**` line, if any.
    pub status: Option<String>,
    pub has_if_wrong: bool,
    pub line: usize,
}

/// Every target directory under `root`'s `targets/` and `fb/targets/` (the latter is absent
/// outside the internal tree), sorted by path.
pub fn discover(root: &Path) -> Result<Vec<Target>> {
    let mut out = Vec::new();
    for d in DIRS {
        let base = root.join(d);
        if !base.is_dir() {
            continue;
        }
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&base)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let f = dir.join("target.toml");
            let text = std::fs::read_to_string(&f)
                .with_context(|| format!("{}: missing target.toml", dir.display()))?;
            let spec: TargetSpec =
                toml::from_str(&text).with_context(|| format!("{}", f.display()))?;
            out.push(Target { dir, spec });
        }
    }
    Ok(out)
}

/// The entries of an assumption registry in the top-level format.
pub fn parse_assumptions(text: &str) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if let Some(h) = l.strip_prefix("### ") {
            if let Some((id, kind)) = h.split_once(" · ") {
                out.push(Entry {
                    id: id.trim().to_string(),
                    kind: kind.trim().to_string(),
                    status: None,
                    has_if_wrong: false,
                    line: i + 1,
                });
            }
            continue;
        }
        let Some(e) = out.last_mut() else { continue };
        if let Some(s) = l.trim_start().strip_prefix("- Status: **") {
            if e.status.is_none() {
                e.status = s.split("**").next().map(str::to_string);
            }
        } else if l.trim_start().starts_with("- If wrong:") {
            e.has_if_wrong = true;
        }
    }
    out
}

/// Consistency problems across all targets; empty when the tree is consistent. `top_ids` are
/// the IDs of the top-level ASSUMPTIONS.md (GB300), which no target may reuse.
pub fn check(root: &Path, top_ids: &[String]) -> Result<Vec<String>> {
    let targets = discover(root)?;
    let mut errs = Vec::new();
    let mut owner: BTreeMap<String, String> = top_ids
        .iter()
        .map(|id| (id.clone(), "the top-level ASSUMPTIONS.md".to_string()))
        .collect();
    let mut ids: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut prefixes: BTreeMap<String, String> = BTreeMap::new();
    for t in &targets {
        let s = &t.spec;
        let name = t.dir.display().to_string();
        let dir_name = t.dir.file_name().unwrap().to_string_lossy().to_string();
        if s.id != dir_name {
            errs.push(format!("{name}: id {:?} is not the directory name", s.id));
        }
        if let Some(prev) = ids.insert(s.id.clone(), t.dir.clone()) {
            errs.push(format!(
                "{name}: id {:?} also used by {}",
                s.id,
                prev.display()
            ));
        }
        if !STATUSES.contains(&s.status.as_str()) {
            errs.push(format!("{name}: status {:?} not in {STATUSES:?}", s.status));
        }
        if !t.assumptions_path().is_file() {
            errs.push(format!(
                "{name}: missing {}",
                t.assumptions_path().display()
            ));
        }
        if !t.spec_path().is_file() {
            errs.push(format!("{name}: missing {}", t.spec_path().display()));
        }
        if t.is_template() || s.id_prefix.is_empty() {
            // The template, and GB300, whose registry is checked by the sim crate's tests.
            continue;
        }
        if s.id_prefix != s.id_prefix.to_ascii_uppercase() || s.id_prefix.contains('-') {
            errs.push(format!(
                "{name}: id_prefix {:?} must be upper case, without '-'",
                s.id_prefix
            ));
        }
        if let Some(prev) = prefixes.insert(s.id_prefix.clone(), s.id.clone()) {
            errs.push(format!(
                "{name}: id_prefix {:?} also used by {prev}",
                s.id_prefix
            ));
        }
        if !t.dir.join("probes").is_dir() {
            errs.push(format!("{name}: missing probes/"));
        }
        let text = std::fs::read_to_string(t.assumptions_path()).unwrap_or_default();
        for e in parse_assumptions(&text) {
            let at = format!("{}:{}", t.assumptions_path().display(), e.line);
            let mut parts = e.id.splitn(3, '-');
            let (p, c, rest) = (parts.next(), parts.next(), parts.next());
            if p != Some(s.id_prefix.as_str()) || rest.is_none_or(str::is_empty) {
                errs.push(format!(
                    "{at}: {} is not {}-<CLASS>-<NAME>",
                    e.id, s.id_prefix
                ));
            } else if !c.is_some_and(|c| CLASSES.contains(&c)) {
                errs.push(format!("{at}: {} has a class not in {CLASSES:?}", e.id));
            }
            if !KINDS.contains(&e.kind.as_str()) {
                errs.push(format!("{at}: kind {:?} not in {KINDS:?}", e.kind));
            }
            match &e.status {
                Some(st) if ASSUMPTION_STATUSES.contains(&st.as_str()) => {}
                st => errs.push(format!(
                    "{at}: {} needs `- Status: **<{}>**`, found {st:?}",
                    e.id,
                    ASSUMPTION_STATUSES.join("|")
                )),
            }
            if !e.has_if_wrong {
                errs.push(format!("{at}: {} needs a `- If wrong:` line", e.id));
            }
            if let Some(prev) = owner.insert(e.id.clone(), name.clone()) {
                errs.push(format!("{at}: {} also defined in {prev}", e.id));
            }
        }
    }
    for t in &targets {
        if let Some(b) = &t.spec.based_on {
            if !ids.contains_key(b) {
                errs.push(format!(
                    "{}: based_on {b:?} is not a target",
                    t.dir.display()
                ));
            }
        }
    }
    Ok(errs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_entries() {
        let doc = "# T\n### H100-S-X · spec\n**A.**\n- Status: **argued** (2026-11-09)\n\
                   - If wrong: miss.\n### H100-E-Y · code\n**B.**\n";
        let e = parse_assumptions(doc);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].id, "H100-S-X");
        assert_eq!(e[0].status.as_deref(), Some("argued"));
        assert!(e[0].has_if_wrong);
        assert_eq!(e[1].status, None);
        assert!(!e[1].has_if_wrong);
    }

    #[test]
    fn rejects_bad_entries() {
        let dir = std::env::temp_dir().join(format!("ptxv-targets-{}", std::process::id()));
        let t = dir.join("targets/x1");
        std::fs::create_dir_all(t.join("probes")).unwrap();
        std::fs::write(
            t.join("target.toml"),
            "id = \"x1\"\nname = \"X\"\nvendor = \"v\"\nisa = []\narch = []\n\
             id_prefix = \"X1\"\nstatus = \"spec\"\nbased_on = \"nope\"\n",
        )
        .unwrap();
        std::fs::write(t.join("SPEC.md"), "# S\n").unwrap();
        std::fs::write(
            t.join("ASSUMPTIONS.md"),
            "### X1-S-OK · spec\n- Status: **validated**\n- If wrong: miss\n\
             ### X2-S-WRONG · spec\n- Status: **validated**\n- If wrong: miss\n\
             ### X1-Q-CLASS · spec\n- Status: **maybe**\n\
             ### X1-E-TOP · spec\n- Status: **argued**\n- If wrong: miss\n",
        )
        .unwrap();
        let errs = check(&dir, &["X1-E-TOP".to_string()]).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let has = |s: &str| errs.iter().any(|e| e.contains(s));
        assert!(has("X2-S-WRONG is not X1"), "{errs:#?}");
        assert!(has("X1-Q-CLASS has a class"), "{errs:#?}");
        assert!(has("found Some(\"maybe\")"), "{errs:#?}");
        assert!(has("X1-Q-CLASS needs a `- If wrong:`"), "{errs:#?}");
        assert!(has("X1-E-TOP also defined"), "{errs:#?}");
        assert!(has("based_on \"nope\""), "{errs:#?}");
        assert!(!has("X1-S-OK"), "{errs:#?}");
    }
}
