/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv errdist`: the distribution of a kernel's rounding error (C - exact) or of a pair's
//! difference (C - R) under a user-given input distribution (docs/errdist.md).
//!
//! These are distributional estimates with stated error bars, not worst-case proofs; the
//! worst-case bound of `ptxv equiv` stays the guarantee.

pub mod cf;
pub mod chain;
pub mod combine;
pub mod errstate;
pub mod fine;
pub mod is;
pub mod laws;
pub mod markov;
pub mod mc;
// Agent AG: non-chain ops (softmax, norms, attention, elementwise).
pub mod nonlin;
pub mod report;
pub mod schema;
pub mod spec;
pub mod special;
pub mod stats;
pub mod tier0;
pub mod tier1;
pub mod tier2;
pub mod tier2bs;
pub mod validate;
