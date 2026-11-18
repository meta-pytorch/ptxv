/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::memory_safety::Event;

use crate::exec::Loc;
use crate::machine::Alloc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedGlobalEvent {
    event: Event,
    location: Loc,
}

impl ResolvedGlobalEvent {
    pub(crate) fn event(self) -> Event {
        self.event
    }

    pub(crate) fn location(self) -> Loc {
        self.location
    }
}

fn resolve_allocation(
    event: Event,
    index: usize,
    base: u64,
    bytes: usize,
) -> Option<ResolvedGlobalEvent> {
    let Event::Access { request, .. } = event else {
        return None;
    };
    let end = request.address as u128 + request.width as u128;
    let allocation_end = base as u128 + bytes as u128;
    if allocation_end <= u64::MAX as u128 + 1 && request.address >= base && end <= allocation_end {
        Some(ResolvedGlobalEvent {
            event,
            location: Loc::Global {
                alloc: index,
                off: request.address - base,
            },
        })
    } else {
        None
    }
}

pub(crate) fn resolve_global_event(
    event: Event,
    allocations: &[Alloc],
) -> Option<ResolvedGlobalEvent> {
    let Event::Access { request, .. } = event else {
        return None;
    };
    if !matches!(request.width, 1 | 2 | 4 | 8)
        || request.alignment != request.width
        || request.address % request.alignment != 0
    {
        return None;
    }
    let mut index = 0;
    let mut found = None;
    while index < allocations.len() && found.is_none() {
        found = resolve_allocation(
            event,
            index,
            allocations[index].base,
            allocations[index].init.len(),
        );
        index += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use ptxv_memory_write::memory_safety::AccessKind;
    use ptxv_memory_write::memory_safety::Request;

    use super::*;

    fn allocation(base: u64) -> Alloc {
        Alloc {
            name: "event".into(),
            base,
            init: std::sync::Arc::new(vec![0; 16]),
            undef: false,
            dead: false,
        }
    }

    #[test]
    fn preserves_actor_descriptor_and_physical_location() {
        for kind in [AccessKind::Load, AccessKind::Store { defined: true }] {
            let event = Event::Access {
                thread: 37,
                request: Request {
                    address: 4104,
                    width: 8,
                    alignment: 8,
                    kind,
                },
            };
            let resolved = resolve_global_event(event, &[allocation(4096)]).unwrap();
            assert_eq!(resolved.event(), event);
            assert_eq!(resolved.location(), Loc::Global { alloc: 0, off: 8 });
        }
    }

    #[test]
    fn widened_resolution_refuses_wrapping_and_misaligned_ranges() {
        for (address, width) in [(u64::MAX - 7, 8), (4105, 8), (4112, 8)] {
            let event = Event::Access {
                thread: 0,
                request: Request {
                    address,
                    width,
                    alignment: width,
                    kind: AccessKind::Load,
                },
            };
            assert!(resolve_global_event(event, &[allocation(4096)]).is_none());
        }
        assert!(resolve_global_event(Event::Control, &[allocation(4096)]).is_none());
        assert!(
            resolve_global_event(
                Event::Access {
                    thread: 0,
                    request: Request {
                        address: u64::MAX - 7,
                        width: 8,
                        alignment: 8,
                        kind: AccessKind::Load,
                    },
                },
                &[allocation(u64::MAX - 7)],
            )
            .is_none()
        );
    }
}
