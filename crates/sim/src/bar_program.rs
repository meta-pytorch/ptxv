/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use ptxv_ir::control_flow;
use ptxv_ir::destination_rows;
use ptxv_ir::register_owner;
use ptxv_memory_write::cache;
use ptxv_memory_write::cache_graph;
use ptxv_memory_write::graph_source;
use ptxv_memory_write::register_index;
use ptxv_memory_write::source_rows;
use ptxv_memory_write::symbolic;

fn register_key(register: Reg) -> symbolic::RegisterKey {
    let (owner, file, index) = register_owner::canonical_key(0, &[], register);
    symbolic::register_key(owner, file, index)
}

struct LegacyDestinations;

impl destination_rows::LegacyDestinations for LegacyDestinations {
    fn row(&self, operation: &Op) -> Vec<Reg> {
        operation.dst_regs()
    }
}

#[derive(Debug, Default)]
pub(crate) struct Flags {
    pub arv_repeat: u32,
    pub arv_mixed: u32,
    pub regid: bool,
    pub cut: u32,
}

pub(crate) struct ProgramBuilder {
    kernel: Arc<Kernel>,
    graph: cache_graph::MayWriteGraph,
    fold: Vec<Option<bool>>,
    rets: Vec<Vec<usize>>,
}

impl ProgramBuilder {
    pub(crate) fn new(kernel: Arc<Kernel>, fold: Vec<Option<bool>>, rets: Vec<Vec<usize>>) -> Self {
        let source = control_flow::graph_and_seeds(&kernel.code, 0, &fold, &rets);
        let graph = graph_source::build(source.successors, source.seeds);
        Self {
            kernel,
            graph,
            fold,
            rets,
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_successors(&mut self) {
        for successors in &mut self.graph.succ {
            successors.clear();
        }
    }

    pub(crate) fn successors(&self) -> &[Vec<usize>] {
        &self.graph.succ
    }

    pub(crate) fn reach(&self) -> &[u32] {
        &self.graph.reach
    }

    pub(crate) fn repeated_arrivals(&self) -> u32 {
        let mut repeated = 0;
        for (pc, instruction) in self.kernel.code.iter().enumerate() {
            if let Op::BarSync {
                id, arrive: true, ..
            } = &instruction.op
            {
                let bit = match id {
                    Val::Imm(value) if *value < 32 => 1u32 << value,
                    _ => u32::MAX,
                };
                let after = self.successors()[pc]
                    .iter()
                    .fold(0, |mask, &next| mask | self.reach()[next]);
                if after & bit != 0 {
                    repeated |= bit;
                }
            }
        }
        repeated
    }
}

/// Static inputs of the barrier analyses, published through a read-only owner.
#[derive(Debug, Default)]
pub struct Inputs {
    pub graph: cache_graph::MayWriteGraph,
    /// Branches decided by launch constants.
    pub fold: Vec<Option<bool>>,
    /// Per pc, the possible return destinations.
    pub rets: Vec<Vec<usize>>,
    pub arv_repeat: u32,
    pub arv_mixed: u32,
    pub regid: bool,
    /// Named all-thread barriers used by E-BAR-CUT.
    pub cut: u32,
}

impl std::ops::Deref for Inputs {
    type Target = cache_graph::MayWriteGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

#[derive(Debug)]
struct Version {
    generation: u64,
    kernel: Option<Arc<Kernel>>,
    inputs: Inputs,
    destinations: Vec<Vec<Reg>>,
    register_index: register_index::RegisterIndex,
}

/// Clones share one immutable program and identity; there is no mutable accessor.
#[derive(Clone, Debug)]
pub(crate) struct BarProgram(Arc<Version>);

impl BarProgram {
    pub(crate) fn new(builder: ProgramBuilder, flags: Flags) -> Self {
        let rows = destination_rows::collect(&builder.kernel.code, &LegacyDestinations);
        Self::publish(
            Some(builder.kernel),
            Inputs {
                graph: builder.graph,
                fold: builder.fold,
                rets: builder.rets,
                arv_repeat: flags.arv_repeat,
                arv_mixed: flags.arv_mixed,
                regid: flags.regid,
                cut: flags.cut,
            },
            rows,
        )
    }

    fn publish(kernel: Option<Arc<Kernel>>, mut inputs: Inputs, rows: Vec<Vec<Reg>>) -> Self {
        assert!(
            kernel.is_some() || inputs.graph.reach.is_empty() || cfg!(test),
            "nonempty barrier analysis program must own its kernel"
        );
        assert_eq!(
            inputs.graph.reach.len(),
            rows.len(),
            "barrier destination graph shape differs"
        );
        if let Some(kernel) = &kernel {
            assert_eq!(
                kernel.code.len(),
                inputs.graph.reach.len(),
                "barrier graph differs from its kernel"
            );
        }
        let raw_keys = destination_rows::keys(&rows);
        let keys = source_rows::from_raw(raw_keys);
        let (register_index, graph) = register_index::publish_graph(inputs.graph, &keys);
        inputs.graph = graph;
        assert!(
            cache_graph::valid_graph(&inputs.graph),
            "invalid barrier analysis graph"
        );
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let generation = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, cache::next_generation)
            .expect("barrier analysis cache program identities exhausted");
        Self(Arc::new(Version {
            generation,
            kernel,
            inputs,
            destinations: rows,
            register_index,
        }))
    }

    pub(crate) fn dsts(&self) -> &[Vec<Reg>] {
        &self.0.destinations
    }

    pub(crate) fn kernel_option(&self) -> Option<&Kernel> {
        self.0.kernel.as_deref()
    }

    pub(crate) fn kernel(&self) -> &Kernel {
        self.0
            .kernel
            .as_ref()
            .expect("nonempty barrier analysis program must own its kernel")
    }

    pub(crate) fn cache_key(&self, pc: u32, bit: u32) -> cache::CacheKey {
        cache::make_key(self.0.generation, pc, bit)
    }

    pub(crate) fn may_write_sets(&self, start: usize, bit: u32) -> cache_graph::MayWriteSets {
        cache_graph::may_write_sets(&self.0.inputs.graph, start, bit)
    }

    pub(crate) fn may_write(
        &self,
        sets: &cache_graph::MayWriteSets,
        pc: usize,
        register: &Reg,
    ) -> bool {
        cache_graph::may_at(sets, pc, self.0.register_index.get(register_key(*register)))
    }
}

impl Default for BarProgram {
    fn default() -> Self {
        Self::publish(None, Inputs::default(), Vec::new())
    }
}

impl std::ops::Deref for BarProgram {
    type Target = Inputs;

    fn deref(&self) -> &Self::Target {
        &self.0.inputs
    }
}

#[cfg(test)]
mod tests {
    use ptxv_ir::File;

    use super::*;

    #[test]
    fn program_constructor_derives_destinations_from_owned_kernel() {
        let source = r#"
.version 8.0
.target sm_90
.address_size 64
.visible .entry selected_rows() {
    .reg .b64 %rd;
    .reg .b32 %r;
    ld.global.u32 %r, [%rd];
    bar.sync 0;
    ret;
}
"#;
        let kernel = Arc::new(ptxv_ptx::parse_kernel(source, Some("selected_rows")).unwrap());
        let count = kernel.code.len();
        let destination = kernel.code[0].op.dst_regs()[0];
        let builder = ProgramBuilder::new(kernel.clone(), Vec::new(), Vec::new());
        let successor_allocation = builder.successors().as_ptr();
        let reach_allocation = builder.reach().as_ptr();
        assert_eq!(builder.successors(), &[vec![1], vec![2], vec![]]);
        assert_eq!(builder.reach(), &[1, 1, 0]);
        let program = BarProgram::new(
            builder,
            Flags {
                arv_repeat: 2,
                arv_mixed: 4,
                regid: true,
                cut: 0,
            },
        );
        assert_eq!(program.succ.as_ptr(), successor_allocation);
        assert_eq!(program.reach.as_ptr(), reach_allocation);
        assert_eq!(program.succ, vec![vec![1], vec![2], vec![]]);
        assert_eq!(program.reach, vec![1, 1, 0]);
        assert_eq!(program.arv_repeat, 2);
        assert_eq!(program.arv_mixed, 4);
        assert!(program.regid);
        assert!(std::ptr::eq(program.kernel(), kernel.as_ref()));
        assert_eq!(program.dsts().len(), count);
        for (row, instruction) in program.dsts().iter().zip(&kernel.code) {
            assert_eq!(*row, instruction.op.dst_regs());
        }
        let sets = program.may_write_sets(0, 1);
        assert!(program.may_write(&sets, 1, &destination));
    }

    #[test]
    fn default_program_has_consistent_empty_graph() {
        let program = BarProgram::default();
        assert!(program.0.kernel.is_none());
        assert!(program.dsts().is_empty());
        assert!(program.succ.is_empty());
        assert!(program.reach.is_empty());
        assert!(program.writes.is_empty());
        assert_eq!(program.register_count, 0);
    }

    #[test]
    fn program_graph_and_query_preserve_register_file_and_full_index() {
        let local = Reg {
            file: File::R,
            idx: 7,
        };
        let uniform = Reg {
            file: File::UR,
            idx: 7,
        };
        let high = Reg {
            file: File::UP,
            idx: u16::MAX,
        };
        let program = BarProgram::publish(
            None,
            Inputs {
                graph: cache_graph::MayWriteGraph {
                    reach: vec![1, 1, 0],
                    succ: vec![vec![1], vec![2], vec![]],
                    writes: vec![],
                    register_count: 0,
                },
                ..Inputs::default()
            },
            vec![vec![local, high], vec![uniform], vec![]],
        );
        let sets = program.may_write_sets(0, 1);
        assert!(program.may_write(&sets, 1, &local));
        assert!(program.may_write(&sets, 1, &high));
        assert!(!program.may_write(&sets, 1, &uniform));
        assert!(program.may_write(&sets, 2, &uniform));
        for file in [File::V, File::P, File::UP] {
            assert!(!program.may_write(&sets, 1, &Reg { file, idx: 7 }));
        }
        assert!(!program.may_write(
            &sets,
            1,
            &Reg {
                file: File::UP,
                idx: 63
            }
        ));
        assert!(!program.may_write(
            &sets,
            1,
            &Reg {
                file: File::UP,
                idx: 32767
            }
        ));
    }
}
