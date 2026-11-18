/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::atomic_components as policy;
use ptxv_memory_write::atomic_components::Component;
use ptxv_memory_write::atomic_components::FrozenOperands;
use ptxv_memory_write::atomic_components::Plan;
use ptxv_memory_write::atomic_packed as packed;
use ptxv_memory_write::atomic_scalar as scalar;

use super::*;

pub(crate) const COMPONENTS: [Component; 4] = [
    Component::First,
    Component::Second,
    Component::Third,
    Component::Fourth,
];

/// Cause 2 (`Machine::lane_orders_exact`): up to this many active lanes, an unobserved scalar
/// atomic whose lane orders are bounded (MISS 74: lane order and its reverse beyond four
/// lanes whose order matters) keeps its components, whose search is complete (eight lanes
/// taking tickets: 190,243 states at the launch check); beyond it the components pass the
/// state cap (ten lanes: 2M states), and the bounded orders stand, with their note.
const LANE_ORDER_COMPONENTS_MAX: usize = 8;

#[derive(Clone, Debug, Hash)]
struct Lane {
    thread: usize,
    input: FrozenOperands,
    plan: Plan,
    coverage_mask: u64,
    packed: Option<Box<PackedLane>>,
}

#[derive(Clone, Debug, Hash)]
struct PackedLane {
    source: [crate::num::Bf16Half; 2],
    returned: packed::Returns,
    provenance: crate::halfin::Bytes,
    read_tainted: bool,
    coverage_mask: u64,
}

#[derive(Clone, Debug, Hash)]
pub(crate) struct Continuation {
    members: Vec<usize>,
    lanes: Vec<Lane>,
    taint: (bool, bool),
}

#[derive(Clone, Copy)]
enum Layout {
    FloatWords(u8),
    PackedBf16,
    ScalarWord(Plan),
}

impl Layout {
    fn plan(self) -> Plan {
        match self {
            Self::FloatWords(bytes) => policy::vector_plan(bytes).expect("vector atomic width"),
            Self::PackedBf16 => packed::plan(),
            Self::ScalarWord(plan) => plan,
        }
    }
}

fn layout(front: Front, op: &Op) -> Option<Layout> {
    if let Op::Atom {
        op,
        bytes,
        d,
        p,
        new,
        ..
    } = op
    {
        if let Some(plan) = scalar::plan(
            front == Front::Ptx,
            operation(*op),
            *bytes,
            d.is_some(),
            p.is_some(),
            new.is_some(),
        ) {
            return Some(Layout::ScalarWord(plan));
        }
    }
    match op {
        Op::Atom {
            op: AtomOp::FAddFtz,
            bytes: bytes @ (8 | 16),
            d: Some(_),
            v: Val::Reg(..),
            p: None,
            new: None,
            ..
        } => Some(Layout::FloatWords(*bytes)),
        Op::Atom {
            op: AtomOp::AddBf16x2,
            bytes: 4,
            d: Some(_),
            p: None,
            new: None,
            ..
        } => Some(Layout::PackedBf16),
        _ => None,
    }
}

impl Machine {
    pub(crate) fn note_unmodeled_scalar_lane_order(&mut self, members: &[usize], pc: u32) {
        let kernel = self.k.clone();
        let instruction = &kernel.code[pc as usize];
        let Op::Atom {
            op,
            bytes,
            d,
            addr,
            space,
            p,
            ..
        } = &instruction.op
        else {
            return;
        };
        let continued = layout(self.k.front, &instruction.op).is_some();
        let monitored = ptxv_ir::atomic_result::local_order_monitored(self.k.front, *op, *d, *p);
        let predicate = ptxv_ir::register_owner::normalize_predicate(
            self.k.front,
            Pred {
                reg: *p,
                neg: false,
            },
        )
        .reg;
        let scalar_red = d.is_none() && *bytes == 4 && *op == AtomOp::FAddFtz;
        if (!scalar_red && !monitored) || continued {
            return;
        }
        let shared_result = d.is_some_and(|d| matches!(d.file, File::UR | File::UP))
            || predicate.is_some_and(|p| matches!(p.file, File::UR | File::UP));
        let claims: Vec<scalar::LaneClaim> = members
            .iter()
            .copied()
            .filter(|&thread| self.prd(thread, instruction.guard))
            .map(|thread| {
                let location = self
                    .addr(thread, addr)
                    .ok()
                    .filter(|&address| check_aligned(*space, address, *bytes as usize).is_ok())
                    .and_then(|address| {
                        self.resolve(thread, *space, address, *bytes as usize).ok()
                    });
                let (region, owner, offset) = match location {
                    Some(Loc::Global { alloc, off }) => (1, alloc, off),
                    Some(Loc::Shared { cta, off }) => (2, cta as usize, off as u64),
                    _ => (0, 0, 0),
                };
                (self.threads[thread].warp, region, owner, offset)
            })
            .collect();
        let unmodeled = scalar::local_claims_unmodeled(
            monitored,
            scalar_red,
            self.wlog.0.is_some(),
            continued,
            shared_result,
            &claims,
        );
        if unmodeled {
            self.report("unmodeled",format!("atomic-lane-order:{pc}"),
                format!("scalar atomic at {} has overlapping or shared-return warp-lane claims whose interleavings are not modeled",self.k.describe(pc as usize)));
            self.rec_fault();
        }
    }

    pub(crate) fn atomic_scalar_supported(&self, pc: u32) -> bool {
        matches!(
            layout(self.k.front, &self.k.code[pc as usize].op),
            Some(Layout::ScalarWord(_))
        )
    }

    /// No search mode that gives AtomicComponent actions a meaning of their own is on:
    /// numerics, the certificate log, C-TICKET-ADDR reruns, global-ownership runs, C-RING /
    /// C-TICKET forcing, input-coverage provenance, the hardware L1, S-INIT fills.
    pub(crate) fn plain_search(&self) -> bool {
        self.num.is_none()
            && !self.wlog.is_on()
            && self.perm.is_none()
            && self.gown.is_none()
            && self.ring.is_none()
            && self.prov.is_none()
            && self.l1.is_none()
            && self.fill.is_none()
    }

    /// Cause 2: may an actor other than `members` access the memory of the atomic at `pc`
    /// before its last lane retires? Conservative: a live thread counts when the control
    /// flow from its pc (an unlaunched one's: its grid's entry) may reach any access of the
    /// atomic's memory class, global or shared (`BarCfg::access`: weak accesses, a trap, and
    /// every instruction not known to be memory-free, count), or when copies it issued have
    /// yet to land (`Thread::cp_pending`: its exit lands them, E-CPASYNC-EXIT). So do a
    /// pending asynchronous operation (a copy landing through an mbarrier arrive after its
    /// issuer has left) and a NIC.
    pub(crate) fn lane_order_observable(&self, members: &[usize], pc: u32) -> bool {
        let access = self.bar_cfg.access();
        let class = match &self.k.code[pc as usize].op {
            Op::Atom { space, .. } => crate::explore::space_access(*space),
            _ => return true,
        };
        if class == 0
            || access.len() != self.k.code.len()
            || !self.pending.is_empty()
            || self.nic.is_some()
            || self.launch.nic.is_some()
        {
            return true;
        }
        self.live_threads().any(|(t, thread)| {
            let from = match thread.st {
                St::Exited => return false,
                St::Unlaunched => self.launch.g(u32::from(thread.cta)).entry,
                _ => thread.pc,
            };
            !members.contains(&t)
                && (!thread.cp_pending.is_empty()
                    || access.get(from as usize).is_none_or(|&a| a & class != 0))
        })
    }

    /// Cause 2 (one mechanism per instruction): PTX, a plain search, no other actor can access
    /// the memory of the atomic at `pc` before its last lane retires
    /// (`lane_order_observable`), and the lanes' orders give what the components give
    /// (`lane_orders_exact`). The members wait at `pc` while its lanes run, and each lane's
    /// step touches only its words, its registers and its clock, so it commutes with every
    /// other actor's step: every interleaving reaches what an order of the lanes run as one
    /// step reaches, and `lane_orders` takes the orders that may differ.
    pub(crate) fn lane_orders_suffice(&self, members: &[usize], pc: u32) -> bool {
        self.k.front == Front::Ptx
            && self.plain_search()
            && !self.lane_order_observable(members, pc)
            && self.lane_orders_exact(members, pc)
    }

    /// Cause 2: where the lane orders of the atomic at `pc` reach exactly what its components
    /// reach. Every active lane's address computes, is aligned and names memory (a fault
    /// stops the lane-order step at its lane, while the components' issue faults before any
    /// lane runs, or a faulting component leaves the other lanes theirs); every active lane's
    /// operand is defined (the components carry an undefined operand into the word and the
    /// returned values, S-INIT); and `lane_orders` takes every order that may differ, unless
    /// more than `LANE_ORDER_COMPONENTS_MAX` lanes are active (MISS 74's bound, with its note,
    /// where the components would pass any state cap).
    fn lane_orders_exact(&self, members: &[usize], pc: u32) -> bool {
        let inst = &self.k.code[pc as usize];
        let Op::Atom {
            space,
            bytes,
            addr,
            v,
            ..
        } = &inst.op
        else {
            return false;
        };
        let active: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&t| self.prd(t, inst.guard))
            .collect();
        for &t in &active {
            let Ok(address) = self.addr(t, addr) else {
                return false;
            };
            if check_aligned(*space, address, *bytes as usize).is_err() {
                return false;
            }
            if self
                .mem_locs(t, *space, address, *bytes as usize, pc)
                .is_err()
            {
                return false;
            }
            if let Val::Reg(operand, _) = v {
                if self.reg_umask(t, *operand) != 0 {
                    return false;
                }
            }
        }
        self.lane_orders(members, pc)
            .is_none_or(|o| o.truncated.is_none() || active.len() > LANE_ORDER_COMPONENTS_MAX)
    }

    pub(crate) fn atomic_scalar_lane_retired(&self, thread: usize, pc: u32) -> bool {
        self.k.code.get(pc as usize).is_some()
            && self.atomic_scalar_supported(pc)
            && self
                .atomic_components
                .get(&(self.threads[thread].warp, pc))
                .and_then(|pending| pending.lanes.iter().find(|lane| lane.thread == thread))
                .is_some_and(|lane| policy::retired(lane.plan))
    }

    pub(crate) fn atomic_scalar_frozen_address(&self, thread: usize, pc: u32) -> Option<u64> {
        if !self.atomic_scalar_supported(pc) {
            return None;
        }
        self.atomic_components
            .get(&(self.threads[thread].warp, pc))?
            .lanes
            .iter()
            .find(|lane| lane.thread == thread)
            .map(|lane| lane.input.address)
    }

    pub(crate) fn atomic_scalar_ticket_progress(&self, members: &[usize], pc: u32) -> bool {
        let Some(&thread) = members.first() else {
            return false;
        };
        let warp = self.threads[thread].warp;
        let active = match self.atomic_components.get(&(warp, pc)) {
            Some(pending) => pending.lanes.iter().any(|lane| !policy::retired(lane.plan)),
            None => members
                .iter()
                .any(|&thread| self.prd(thread, self.k.code[pc as usize].guard)),
        };
        scalar::ticket_progress(
            active,
            self.atomic_component_choices(warp, pc, members).is_some(),
        )
    }
    pub(crate) fn atomic_component_members(&self, warp: u32, pc: u32) -> Vec<usize> {
        self.atomic_components
            .get(&(warp, pc))
            .map(|pending| pending.members.clone())
            .unwrap_or_else(|| {
                self.warp_threads(warp)
                    .filter(|&t| self.threads[t].pc == pc && self.threads[t].st == St::Run)
                    .collect()
            })
    }

    pub(crate) fn atomic_component_choices(
        &self,
        warp: u32,
        pc: u32,
        members: &[usize],
    ) -> Option<Vec<(usize, Component)>> {
        let layout = layout(self.k.front, &self.k.code[pc as usize].op)?;
        // Cause 2: one mechanism per instruction. A scalar word (one component per lane)
        // whose lanes' orders suffice takes them (`lane_orders`) instead of components; the
        // choice is made at issue, so a pending continuation keeps its components.
        if matches!(layout, Layout::ScalarWord(_))
            && !self.atomic_components_always
            && !self.atomic_components.contains_key(&(warp, pc))
            && self.lane_orders_suffice(members, pc)
        {
            return None;
        }
        let mut out = Vec::new();
        if let Some(pending) = self.atomic_components.get(&(warp, pc)) {
            for index in 0..pending.lanes.len() {
                let lane = &pending.lanes[if self.atom_lanes_rev {
                    pending.lanes.len() - 1 - index
                } else {
                    index
                }];
                for component in policy::choices(lane.plan) {
                    out.push((lane.thread, component));
                }
            }
        } else {
            let plan = layout.plan();
            for index in 0..members.len() {
                let thread = members[if self.atom_lanes_rev {
                    members.len() - 1 - index
                } else {
                    index
                }];
                if self.prd(thread, self.k.code[pc as usize].guard) {
                    for component in policy::choices(plan) {
                        out.push((thread, component));
                    }
                }
            }
        }
        if matches!(layout, Layout::ScalarWord(_))
            && self
                .ring
                .as_ref()
                .is_some_and(|ring| ring.tick_pcs.contains(&pc))
        {
            out.retain(|&(thread, _)| self.ticket_ready_actor(thread, pc));
        }
        (!out.is_empty()).then_some(out)
    }

    fn issue_atomic_components(
        &mut self,
        members: &[usize],
        pc: u32,
    ) -> Result<Continuation, String> {
        let k = self.k.clone();
        let inst = &k.code[pc as usize];
        let Op::Atom {
            space,
            bytes,
            addr,
            v,
            d,
            ..
        } = &inst.op
        else {
            return Err("component issue requires a vector atomic".into());
        };
        let layout =
            layout(self.k.front, &inst.op).expect("component instruction selected before issue");
        let plan = layout.plan();
        let active: Vec<_> = members
            .iter()
            .copied()
            .filter(|&t| self.prd(t, inst.guard))
            .collect();
        for &thread in members {
            if let Some(key) = self.take_thread_acquire(thread) {
                self.rec_acq(key);
            }
        }
        self.imply_tc_fence(members);
        self.wait_instruction_bulk_reads(members, pc);
        self.consume_instruction_reads(&active, inst);
        self.sinit_pre(members, pc);
        self.taint_pre(members, &active, pc);
        let taint = self.atomic_taint_context();
        let guard_tainted = self.atomic_guard_tainted(members, pc);
        let coverage_masks = crate::coverage::capture_atomic_issue(self, members, pc);
        if self.wlog.is_on() {
            crate::wellsync::pre_inst(self, members, pc);
            crate::wellsync::post_inst(self, members, pc, true);
        }
        let mut lanes = Vec::new();
        for thread in active {
            self.check_thread_inputs(thread, inst, pc)?;
            let address = self.addr(thread, addr)?;
            check_aligned(*space, address, *bytes as usize)?;
            let empty = ptxv_memory_write::symbolic::prepare(0, None, false, true);
            let mut sources = [empty; 4];
            let packed_lane = match layout {
                Layout::FloatWords(_) => {
                    let Val::Reg(source, _) = v else {
                        unreachable!()
                    };
                    for component in COMPONENTS {
                        if policy::pending(plan, component) {
                            let index = policy::byte_offset(component) as usize / 4;
                            let value = Val::Reg(
                                Reg {
                                    file: source.file,
                                    idx: source.idx + index as u16,
                                },
                                1,
                            );
                            let bits = self.rd(thread, &value)? as u32;
                            sources[index] = self.capture_atomic_source(thread, &value, bits);
                        }
                    }
                    None
                }
                Layout::ScalarWord(_) => {
                    let bits = self.rd(thread, v)? as u32;
                    sources[0] = self.capture_atomic_source(thread, v, bits);
                    sources[0].defined = match v {
                        Val::Reg(source, _) => self.reg_umask(thread, *source) == 0,
                        _ => true,
                    };
                    None
                }
                Layout::PackedBf16 => {
                    let word = self.rd(thread, v)? as u32;
                    let mask = match v {
                        Val::Reg(source, _) => self.reg_umask(thread, *source),
                        _ => 0,
                    };
                    let mut frozen = [crate::num::Bf16Half::Unknown; 2];
                    for component in [Component::First, Component::Second] {
                        let index =
                            packed::offset(component).expect("packed component") as usize / 2;
                        let bits = packed::operand(word, component).expect("packed component");
                        let defined =
                            packed::operand_defined(mask, component).expect("packed component");
                        if defined {
                            frozen[index] =
                                self.capture_atomic_bf16_source(thread, v, index as u32);
                        }
                        sources[index] = ptxv_memory_write::symbolic::prepare(
                            bits as u32,
                            None,
                            taint.0,
                            defined,
                        );
                    }
                    Some(Box::new(PackedLane {
                        source: frozen,
                        returned: packed::empty_returns(),
                        provenance: [None; 4],
                        read_tainted: false,
                        coverage_mask: 0,
                    }))
                }
            };
            lanes.push(Lane {
                thread,
                input: FrozenOperands {
                    address,
                    first: sources[0],
                    second: sources[1],
                    third: sources[2],
                    fourth: sources[3],
                },
                plan,
                coverage_mask: coverage_masks.get(&thread).copied().unwrap_or(0),
                packed: packed_lane,
            });
        }
        if let Some(d) = d {
            for &t in members {
                if !lanes.iter().any(|lane| lane.thread == t) && guard_tainted {
                    self.begin_atomic_component_taint(taint);
                    for component in COMPONENTS {
                        if policy::pending(plan, component) {
                            let index = policy::byte_offset(component) as u16 / 4;
                            if matches!(layout, Layout::PackedBf16) && index != 0 {
                                continue;
                            }
                            self.finish_atomic_component_taint(
                                t,
                                Some(Reg {
                                    file: d.file,
                                    idx: d.idx + index,
                                }),
                            );
                        }
                    }
                }
            }
        }
        Ok(Continuation {
            members: members.to_vec(),
            lanes,
            taint,
        })
    }

    pub(crate) fn step_atomic_component(
        &mut self,
        warp: u32,
        pc: u32,
        thread: usize,
        component: Component,
    ) {
        let members: Vec<_> = self
            .warp_threads(warp)
            .filter(|&t| self.threads[t].pc == pc && self.threads[t].st == St::Run)
            .collect();
        let mut continuation = match self.atomic_components.remove(&(warp, pc)) {
            Some(continuation) => continuation,
            None => match self.issue_atomic_components(&members, pc) {
                Ok(continuation) => continuation,
                Err(error) => {
                    self.atomic_component_fault(&members, pc, &error);
                    return;
                }
            },
        };
        let lane = continuation
            .lanes
            .iter_mut()
            .find(|lane| lane.thread == thread)
            .expect("atomic component belongs to issued split");
        let packed_component = lane.packed.is_some();
        let scalar_component = self.atomic_scalar_supported(pc);
        let selected = if packed_component {
            packed::begin_component(lane.input, lane.plan, component)
        } else {
            policy::begin_component(lane.input, lane.plan, component)
        }
        .expect("atomic component is pending");
        let offset = if packed_component {
            packed::offset(component).expect("packed component")
        } else {
            policy::byte_offset(component)
        };
        let address = selected.address;
        let source = selected.source;
        let k = self.k.clone();
        let inst = &k.code[pc as usize];
        let Op::Atom {
            op, space, d, sync, ..
        } = &inst.op
        else {
            unreachable!()
        };
        let destination = d.map(|d| Reg {
            file: d.file,
            idx: d.idx
                + if packed_component {
                    0
                } else {
                    offset as u16 / 4
                },
        });
        let scalar = Inst {
            op: Op::Atom {
                op: if packed_component {
                    AtomOp::AddBf16
                } else if scalar_component {
                    *op
                } else {
                    AtomOp::FAddFtz
                },
                space: *space,
                bytes: if packed_component { 2 } else { 4 },
                addr: Addr {
                    base: None,
                    base2: None,
                    off: address as i64,
                },
                v: Val::Imm(source.captured as u64),
                d: destination,
                sync: *sync,
                p: None,
                new: None,
            },
            guard: Pred::TRUE,
            ..inst.clone()
        };
        let mut capture = if let Some(packed) = lane.packed.as_ref() {
            Some(AtomicReturnCapture {
                source: packed.source[offset as usize / 2],
                source_defined: source.defined,
                defer_return: true,
                returned: None,
            })
        } else if scalar_component {
            Some(AtomicReturnCapture {
                source: crate::num::Bf16Half::Unknown,
                source_defined: source.defined,
                defer_return: false,
                returned: None,
            })
        } else {
            None
        };
        let perm_word = self
            .perm
            .as_ref()
            .and_then(|run| run.claim)
            .filter(|claim| scalar_component && claim.0 == thread as u32 && claim.1 == pc)
            .and_then(|_| self.perm_word(thread, &scalar.op));
        if scalar_component
            && self
                .ring
                .as_ref()
                .is_some_and(|ring| ring.tick_pcs.contains(&pc))
        {
            self.ticket_count_actor_at(thread, *space, address, 4);
        }
        self.rec_pc(warp, pc, true);
        let logged = self.wlog.is_on();
        let covered = self.prov.is_some();
        let original_kernel = if logged || covered {
            let original = self.k.clone();
            let mut view = (*original).clone();
            view.code[pc as usize] = scalar.clone();
            self.k = Arc::new(view);
            Some(original)
        } else {
            None
        };
        if logged {
            crate::wellsync::begin_atomic_component(self);
        }
        if covered {
            crate::coverage::begin_atomic_component(
                self,
                thread,
                lane.coverage_mask,
                pc,
                !packed_component,
            );
        }
        self.begin_atomic_component_taint(continuation.taint);
        self.atomic_operand = Some(source);
        self.stale = None;
        self.vmask = None;
        self.wundef = 0;
        let result = self.exec_thread_inner_captured(thread, &scalar, pc, capture.as_mut());
        if logged {
            crate::wellsync::post_inst(self, &[thread], pc, result.is_ok());
        }
        if covered && result.is_ok() {
            if let Some(packed) = lane.packed.as_mut() {
                packed.coverage_mask |= crate::coverage::capture_atomic_component_return(
                    self,
                    thread,
                    lane.coverage_mask,
                );
            } else {
                crate::coverage::post(self, &[thread], pc);
                crate::coverage::finish_atomic_component(self, thread, pc);
            }
        }
        if let Some(original) = original_kernel {
            self.k = original;
        }
        self.atomic_operand = None;
        self.stale = None;
        self.vmask = None;
        self.wundef = 0;
        if let Err(error) = result {
            self.atomic_component_fault(&continuation.members, pc, &error);
            return;
        }
        lane.plan = selected.remaining;
        if let Some(packed) = lane.packed.as_mut() {
            let returned = capture
                .expect("packed component return sink")
                .returned
                .expect("successful atomic returns a value");
            packed.returned = packed::record(
                packed.returned,
                component,
                returned.bits as u16,
                returned.undefined,
            )
            .expect("packed component");
            packed.provenance[offset as usize..offset as usize + 2]
                .copy_from_slice(&returned.provenance);
            packed.read_tainted |= returned.read_tainted;
            if policy::retired(lane.plan) {
                let destination = destination.expect("returning packed instruction");
                self.vmask = Some(packed::undefined_mask(packed.returned) as u64);
                self.wr(
                    thread,
                    destination,
                    packed::returned(packed.returned) as u64,
                    1,
                );
                self.vmask = None;
                self.finish_atomic_packed_taint(
                    thread,
                    destination,
                    packed.read_tainted,
                    packed.provenance,
                );
                if covered {
                    crate::coverage::retire_atomic_packed_return(
                        self,
                        thread,
                        destination,
                        packed.coverage_mask,
                        pc,
                    );
                }
            }
        } else {
            self.finish_atomic_component_taint(thread, destination);
        }
        if let Some(word) = perm_word {
            self.perm_claim(thread, pc, word, &scalar.op);
        }
        if continuation
            .lanes
            .iter()
            .all(|lane| policy::retired(lane.plan))
        {
            crate::coverage::retire_atomic_components(self, warp, pc);
            for &t in &continuation.members {
                if self.threads[t].st == St::Run {
                    self.threads[t].make_mut().set_pc(pc + 1);
                }
            }
            if self.k.front == Front::Sass && continuation.members.len() > 1 {
                let groups = crate::wellsync::lockstep_groups(self, &continuation.members)
                    .unwrap_or_else(|| vec![continuation.members.clone()]);
                for group in groups {
                    let joined = vc_join_all(group.iter().map(|&t| &self.threads[t].vc));
                    for &t in &group {
                        let state = self.threads[t].make_mut();
                        state.vc = joined.clone();
                        vc_tick(&mut state.vc, t as u32);
                    }
                }
                self.join_lanes_sh(&continuation.members);
            }
            if self.wlog.is_on() {
                crate::wellsync::retire_atomic_components(self, &continuation.members);
            }
            self.note_divergence(warp, &continuation.members);
        } else {
            self.atomic_components.insert((warp, pc), continuation);
        }
    }

    fn atomic_component_fault(&mut self, members: &[usize], pc: u32, error: &str) {
        if let Some(&thread) = members.first() {
            crate::coverage::retire_atomic_components(self, self.threads[thread].warp, pc);
        }
        self.report(
            fault_kind(error),
            format!("fault:{pc}"),
            format!("{error} at {}", self.k.describe(pc as usize)),
        );
        self.rec_fault();
        for &t in members {
            self.threads[t].make_mut().set_status(St::Exited);
        }
    }
}
