/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::Kernel;
use ptxv_memory_write::memory_safety::State;
use serde::Serialize;

use crate::machine::Launch;
use crate::strict_memory_admission;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Safe,
    Unsafe,
    Inconclusive,
}

impl Status {
    pub fn is_safe(self) -> bool {
        match self {
            Self::Safe => true,
            Self::Unsafe | Self::Inconclusive => false,
        }
    }
}

pub fn status_for_certificate(kernel: &Kernel, launch: &Launch, certificate: &[State]) -> Status {
    if strict_memory_admission::verify_certificate(kernel, launch, certificate) {
        Status::Safe
    } else {
        Status::Inconclusive
    }
}
