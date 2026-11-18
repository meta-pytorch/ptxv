/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[derive(Clone, Copy, Debug)]
pub struct RegisterState<'a> {
    pub scalar: &'a [u64],
    pub predicates: &'a [u64],
    pub predicate_defined: &'a [u64],
    pub scalar_defined: &'a [u64],
    pub vector_undefined: &'a [(u16, u64)],
    pub uniform: &'a [u64],
    pub uniform_defined: &'a [u64],
    pub uniform_predicates: u8,
    pub uniform_predicate_defined: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryByte {
    pub bits: u8,
    pub defined: bool,
}

pub fn same_words(before: &[u64], after: &[u64]) -> bool {
    if before.len() != after.len() {
        return false;
    }
    let mut index = 0;
    while index < before.len() {
        if before[index] != after[index] {
            return false;
        }
        index += 1;
    }
    true
}

pub fn same_undefined(before: &[(u16, u64)], after: &[(u16, u64)]) -> bool {
    if before.len() != after.len() {
        return false;
    }
    let mut index = 0;
    while index < before.len() {
        if before[index].0 != after[index].0 || before[index].1 != after[index].1 {
            return false;
        }
        index += 1;
    }
    true
}

pub fn same_registers(before: RegisterState<'_>, after: RegisterState<'_>) -> bool {
    same_words(before.scalar, after.scalar)
        && same_words(before.predicates, after.predicates)
        && same_words(before.predicate_defined, after.predicate_defined)
        && same_words(before.scalar_defined, after.scalar_defined)
        && same_undefined(before.vector_undefined, after.vector_undefined)
        && same_words(before.uniform, after.uniform)
        && same_words(before.uniform_defined, after.uniform_defined)
        && before.uniform_predicates == after.uniform_predicates
        && before.uniform_predicate_defined == after.uniform_predicate_defined
}

pub fn same_memory(before: &[MemoryByte], after: &[MemoryByte]) -> bool {
    if before.len() != after.len() {
        return false;
    }
    let mut index = 0;
    while index < before.len() {
        if before[index].bits != after[index].bits || before[index].defined != after[index].defined
        {
            return false;
        }
        index += 1;
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StutterAudit {
    pub memory_unchanged: bool,
}

pub fn begin_audit() -> StutterAudit {
    StutterAudit {
        memory_unchanged: true,
    }
}

pub fn record_store(
    audit: StutterAudit,
    before: &[MemoryByte],
    after: &[MemoryByte],
) -> StutterAudit {
    StutterAudit {
        memory_unchanged: audit.memory_unchanged && same_memory(before, after),
    }
}

pub fn accepts(audit: StutterAudit, before: RegisterState<'_>, after: RegisterState<'_>) -> bool {
    audit.memory_unchanged && same_registers(before, after)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registers(words: &[u64]) -> RegisterState<'_> {
        RegisterState {
            scalar: words,
            predicates: words,
            predicate_defined: words,
            scalar_defined: words,
            vector_undefined: &[],
            uniform: words,
            uniform_defined: words,
            uniform_predicates: 0,
            uniform_predicate_defined: 0,
        }
    }

    #[test]
    fn every_register_plane_participates_in_stutter_checks() {
        let original = registers(&[0]);
        assert!(accepts(begin_audit(), original, original));
        for changed in [
            RegisterState {
                scalar: &[1],
                ..original
            },
            RegisterState {
                scalar: &[],
                ..original
            },
            RegisterState {
                predicates: &[1],
                ..original
            },
            RegisterState {
                predicate_defined: &[1],
                ..original
            },
            RegisterState {
                scalar_defined: &[1],
                ..original
            },
            RegisterState {
                vector_undefined: &[(0, 1)],
                ..original
            },
            RegisterState {
                uniform: &[1],
                ..original
            },
            RegisterState {
                uniform_defined: &[1],
                ..original
            },
            RegisterState {
                uniform_predicates: 1,
                ..original
            },
            RegisterState {
                uniform_predicate_defined: 1,
                ..original
            },
        ] {
            assert!(!accepts(begin_audit(), original, changed), "{changed:?}");
        }
        assert!(!same_undefined(&[(0, 1)], &[(1, 1)]));
        assert!(!same_undefined(&[(0, 1)], &[(0, 2)]));
    }

    #[test]
    fn restoring_stores_and_equal_undefined_bytes_cannot_hide_changes() {
        let original = [MemoryByte {
            bits: 0,
            defined: true,
        }; 32];
        for lane in 0..original.len() {
            let mut changed = original;
            changed[lane].bits = 1;
            let failed = record_store(begin_audit(), &original, &changed);
            assert!(!failed.memory_unchanged);
            let restored = record_store(failed, &original, &original);
            assert!(!restored.memory_unchanged);
            changed[lane] = MemoryByte {
                bits: 0,
                defined: false,
            };
            assert!(!record_store(begin_audit(), &original, &changed).memory_unchanged);
        }
        assert!(!same_memory(&original, &original[..31]));
        assert!(record_store(begin_audit(), &original, &original).memory_unchanged);
    }
}
