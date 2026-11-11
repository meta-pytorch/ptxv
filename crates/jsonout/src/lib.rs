/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `--json`: one JSON document on stdout, everything else on stderr (docs/json.md).
//!
//! The commands print their human-readable report with `println!` throughout. In JSON mode,
//! file descriptor 1 is pointed at stderr for the duration of the command (child processes
//! inherit that), and the document is written to the saved original stdout at the end.
//! Shared by the `ptxv` binary and other front-end binaries built on the ptxv crates.

use std::io::Write;
use std::os::fd::FromRawFd;

extern "C" {
    fn dup(fd: i32) -> i32;
    fn dup2(a: i32, b: i32) -> i32;
}

/// Version of the JSON documents (bumped on incompatible changes; additions keep it).
pub const SCHEMA: u32 = 1;

/// The ptxv commit this binary was built from (build.rs: `PTXV_REV`, git HEAD of the workspace
/// with "+dirty" for uncommitted source changes, or a `REV` file); None when unknown.
pub fn rev() -> Option<&'static str> {
    let r = env!("PTXV_REV");
    (!r.is_empty()).then_some(r)
}

/// "<crate version> (rev <rev>)" / "<crate version> (rev unknown)", for `--version`.
pub const VERSION: &str = env!("PTXV_VERSION");

pub struct JsonOut {
    saved: i32,
}

/// The saved stdout and command of a JSON-mode command that has not emitted yet: `fail`
/// writes an error document there when the command returns an error.
static PENDING: std::sync::Mutex<Option<(i32, String)>> = std::sync::Mutex::new(None);

/// Start JSON mode for `command`: stdout now goes to stderr.
pub fn begin(command: &str) -> JsonOut {
    let _ = std::io::stdout().flush();
    let saved = unsafe { dup(1) };
    unsafe { dup2(2, 1) };
    *PENDING.lock().unwrap() = Some((saved, command.to_string()));
    JsonOut { saved }
}

/// The command failed: `{"error": ...}` instead of its document (no-op outside JSON mode).
pub fn fail(err: &str) {
    let p = PENDING.lock().unwrap().take();
    if let Some((saved, command)) = p {
        JsonOut { saved }.write(&command, serde_json::json!({ "error": err }));
    }
}

impl JsonOut {
    /// Write the document (with `ptxv` version, `ptxv_rev` and schema stamped in) to the
    /// original stdout. The version is this crate's, i.e. the workspace version every ptxv
    /// crate shares; `ptxv_rev` the build's commit ([`rev`], null when unknown).
    pub fn emit(self, command: &str, v: serde_json::Value) {
        PENDING.lock().unwrap().take();
        self.write(command, v);
    }

    fn write(self, command: &str, mut v: serde_json::Value) {
        let _ = std::io::stdout().flush();
        if let Some(o) = v.as_object_mut() {
            o.insert("command".into(), command.into());
            o.insert("schema".into(), SCHEMA.into());
            o.insert("ptxv".into(), env!("CARGO_PKG_VERSION").into());
            o.insert("ptxv_rev".into(), rev().into());
        }
        let mut f = unsafe { std::fs::File::from_raw_fd(self.saved) };
        let _ = writeln!(f, "{}", serde_json::to_string(&v).unwrap_or_default());
        let _ = f.flush();
        // Restore fd 1 (the File closes the saved descriptor when dropped).
        unsafe { dup2(self.saved, 1) };
    }
}
