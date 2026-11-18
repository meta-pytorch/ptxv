/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Op;
use crate::Reg;
use crate::register_effect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    History {
        thread: usize,
        destination: Reg,
        active: bool,
        guarded: bool,
    },
    PrepareSources {
        is_load: bool,
    },
    ClearBits {
        thread: usize,
        destination: Reg,
        active: bool,
        guarded: bool,
    },
    SetActiveTaint {
        thread: usize,
        destination: Reg,
        guarded: bool,
        source_tainted: bool,
    },
    ForceGuardTaint {
        thread: usize,
        destination: Reg,
        active: bool,
    },
}

pub trait MetadataWriter {
    fn apply(&mut self, action: Action);
    fn source_tainted(&self, thread: usize, destination: Reg) -> bool;
}

fn contains(threads: &[usize], thread: usize) -> bool {
    let mut index = 0;
    while index < threads.len() {
        if threads[index] == thread {
            return true;
        }
        index += 1;
    }
    false
}

fn touched_threads(active: &[usize], guarded: &[usize]) -> Vec<usize> {
    let mut touched = active.to_vec();
    let mut index = 0;
    while index < guarded.len() {
        let thread = guarded[index];
        if !contains(active, thread) {
            touched.push(thread);
        }
        index += 1;
    }
    touched
}

fn history<W: MetadataWriter>(
    destination: Reg,
    touched: &[usize],
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < touched.len() {
        let thread = touched[index];
        let active = contains(active, thread);
        let guarded = contains(guarded, thread);
        writer.apply(Action::History {
            thread,
            destination,
            active,
            guarded,
        });
        index += 1;
    }
}

fn clear_bits<W: MetadataWriter>(
    destination: Reg,
    touched: &[usize],
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < touched.len() {
        let thread = touched[index];
        let active = contains(active, thread);
        let guarded = contains(guarded, thread);
        writer.apply(Action::ClearBits {
            thread,
            destination,
            active,
            guarded,
        });
        index += 1;
    }
}

fn active_taint<W: MetadataWriter>(
    destination: Reg,
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < active.len() {
        let thread = active[index];
        let guarded = contains(guarded, thread);
        let source_tainted = writer.source_tainted(thread, destination);
        writer.apply(Action::SetActiveTaint {
            thread,
            destination,
            guarded,
            source_tainted,
        });
        index += 1;
    }
}

fn guard_taint<W: MetadataWriter>(
    destination: Reg,
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < guarded.len() {
        let thread = guarded[index];
        let active = contains(active, thread);
        writer.apply(Action::ForceGuardTaint {
            thread,
            destination,
            active,
        });
        index += 1;
    }
}

fn histories<W: MetadataWriter>(
    destinations: &[Reg],
    touched: &[usize],
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < destinations.len() {
        history(destinations[index], touched, active, guarded, writer);
        index += 1;
    }
}

fn updates<W: MetadataWriter>(
    destinations: &[Reg],
    touched: &[usize],
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) {
    let mut index = 0;
    while index < destinations.len() {
        let destination = destinations[index];
        clear_bits(destination, touched, active, guarded, writer);
        active_taint(destination, active, guarded, writer);
        guard_taint(destination, active, guarded, writer);
        index += 1;
    }
}

fn is_load(operation: &Op) -> bool {
    matches!(operation, Op::Ld { .. })
}

pub fn execute<W: MetadataWriter>(
    operation: &Op,
    active: &[usize],
    guarded: &[usize],
    writer: &mut W,
) -> bool {
    let Some(destinations) = register_effect::memory_writes(operation) else {
        return false;
    };
    let touched = touched_threads(active, guarded);
    histories(&destinations, &touched, active, guarded, writer);
    let is_load = is_load(operation);
    writer.apply(Action::PrepareSources { is_load });
    updates(&destinations, &touched, active, guarded, writer);
    true
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use super::*;
    use crate::Addr;
    use crate::File;
    use crate::Space;
    use crate::Val;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Event {
        Apply(Action),
        Read(usize, Reg),
    }

    #[derive(Default)]
    struct Capture {
        events: RefCell<Vec<Event>>,
        prepared: bool,
        fail_at: Option<usize>,
    }

    impl MetadataWriter for Capture {
        fn apply(&mut self, action: Action) {
            self.events.borrow_mut().push(Event::Apply(action));
            if self.fail_at == Some(self.events.borrow().len() - 1) {
                panic!("injected callback failure");
            }
            if matches!(action, Action::PrepareSources { .. }) {
                self.prepared = true;
            }
        }

        fn source_tainted(&self, thread: usize, destination: Reg) -> bool {
            assert!(self.prepared, "source queried before preparation");
            self.events
                .borrow_mut()
                .push(Event::Read(thread, destination));
            if self.fail_at == Some(self.events.borrow().len() - 1) {
                panic!("injected query failure");
            }
            (thread + destination.idx as usize) % 2 == 0
        }
    }

    fn expected(
        destinations: &[Reg],
        active: &[usize],
        guarded: &[usize],
        is_load: bool,
    ) -> Vec<Event> {
        let touched: Vec<_> = active
            .iter()
            .copied()
            .chain(
                guarded
                    .iter()
                    .copied()
                    .filter(|thread| !active.contains(thread)),
            )
            .collect();
        let mut events = Vec::new();
        for &destination in destinations {
            for &thread in &touched {
                events.push(Event::Apply(Action::History {
                    thread,
                    destination,
                    active: active.contains(&thread),
                    guarded: guarded.contains(&thread),
                }));
            }
        }
        events.push(Event::Apply(Action::PrepareSources { is_load }));
        for &destination in destinations {
            for &thread in &touched {
                events.push(Event::Apply(Action::ClearBits {
                    thread,
                    destination,
                    active: active.contains(&thread),
                    guarded: guarded.contains(&thread),
                }));
            }
            for &thread in active {
                events.push(Event::Read(thread, destination));
                events.push(Event::Apply(Action::SetActiveTaint {
                    thread,
                    destination,
                    guarded: guarded.contains(&thread),
                    source_tainted: (thread + destination.idx as usize) % 2 == 0,
                }));
            }
            for &thread in guarded {
                events.push(Event::Apply(Action::ForceGuardTaint {
                    thread,
                    destination,
                    active: active.contains(&thread),
                }));
            }
        }
        events
    }

    fn operation() -> Op {
        let base = Reg {
            file: File::UR,
            idx: 8,
        };
        Op::Ld256 {
            hi: base,
            lo: base,
            addr: Addr {
                base: Some(Val::Reg(base, 1)),
                base2: None,
                off: 0,
            },
            sync: None,
            nc: false,
            mask: 0x3c,
        }
    }

    #[test]
    fn exact_delivery_order_with_aliases_guard_only_threads_and_duplicate_members() {
        let registers = [8, 9, 8, 9].map(|idx| Reg {
            file: File::UR,
            idx,
        });
        for (active, guarded) in [
            (vec![], vec![]),
            (vec![0, 1], vec![]),
            (vec![], vec![1, 2]),
            (vec![0, 1], vec![1, 2]),
            (vec![0, 0], vec![0, 1, 1]),
        ] {
            let mut writer = Capture::default();
            assert!(execute(&operation(), &active, &guarded, &mut writer));
            assert_eq!(
                *writer.events.borrow(),
                expected(&registers, &active, &guarded, false)
            );
        }
    }

    #[test]
    fn empty_selected_writes_prepare_once_unsupported_has_no_callbacks() {
        let store = Op::St {
            space: Space::Global,
            bytes: 4,
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            v: Val::Imm(0),
            sync: None,
        };
        let mut writer = Capture::default();
        assert!(execute(&store, &[0, 1], &[1, 2], &mut writer));
        assert_eq!(
            *writer.events.borrow(),
            vec![Event::Apply(Action::PrepareSources { is_load: false })]
        );
        writer.events.borrow_mut().clear();
        assert!(!execute(&Op::Exit, &[0, 1], &[1, 2], &mut writer));
        assert!(writer.events.borrow().is_empty());
    }

    #[test]
    fn ordinary_load_derives_source_preparation_from_actual_variant() {
        let destination = Reg {
            file: File::V,
            idx: 4,
        };
        let load = Op::Ld {
            d: destination,
            n: 2,
            space: Space::Global,
            bytes: 8,
            addr: Addr {
                base: Some(Val::Reg(destination, 1)),
                base2: None,
                off: 0,
            },
            sync: None,
            nc: false,
            sext: false,
        };
        let mut writer = Capture::default();
        assert!(execute(&load, &[0], &[1], &mut writer));
        assert_eq!(
            *writer.events.borrow(),
            expected(
                &[
                    destination,
                    Reg {
                        idx: 5,
                        ..destination
                    }
                ],
                &[0],
                &[1],
                true
            )
        );
    }

    #[test]
    fn every_callback_or_query_failure_stops_at_the_attempted_prefix() {
        let expected = expected(
            &[8, 9, 8, 9].map(|idx| Reg {
                file: File::UR,
                idx,
            }),
            &[0, 1],
            &[1, 2],
            false,
        );
        for fail_at in 0..expected.len() {
            let mut writer = Capture {
                fail_at: Some(fail_at),
                ..Default::default()
            };
            assert!(
                catch_unwind(AssertUnwindSafe(|| execute(
                    &operation(),
                    &[0, 1],
                    &[1, 2],
                    &mut writer
                )))
                .is_err()
            );
            assert_eq!(*writer.events.borrow(), expected[..=fail_at]);
        }
    }
}
