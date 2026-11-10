/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! tpu7x TensorCore target of ptxv: LLO listing front-end (`llo`, `lift`), the bitwise
//! machine (`machine`, numerics hooks in `num`), probe-corpus driver (`corpus`) and the
//! explorer model (`model`). Design: fb/targets/tpu/notes/sim.md.

pub mod binlift;
pub mod corpus;
pub mod data;
pub mod dis;
pub mod exe;
pub mod inflate;
pub mod layout;
pub mod lift;
pub mod llo;
pub mod machine;
pub mod model;
pub mod num;
pub mod optab;
