# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The case writer's pointer-table word scan (_casewriter._ptr_words) against the per-word loop
it replaces: the same words, in the same order, on random tables and component layouts
(adjacent, empty and far-apart components; words below 0x10000, at or above 2^52, repeated, at
a component's end, and a trailing partial word). No GPU."""

import random
import struct

import numpy as np
from ptxv._casewriter import _ptr_words


def comp_of(comps, v):
    """The writer's comp_of: the component holding v, else one that ends at v."""
    for i, (s, e) in enumerate(comps):
        if s <= v < e:
            return i
    for i, (s, e) in enumerate(comps):
        if v == e:
            return i
    return None


def loop_words(data, comps):
    """The loop _ptr_words replaces: each new word comp_of resolves, in order."""
    seen, out = set(), []
    for o in range(0, len(data) - 7, 8):
        v = struct.unpack_from("<Q", data, o)[0]
        if v < 0x10000 or v >> 52 or v in seen:
            continue
        seen.add(v)
        if comp_of(comps, v) is not None:
            out.append(v)
    return out


def layout(rng):
    """Sorted, disjoint components as the writer builds them (empty ones included); some
    layouts also hold a component across 0x10000 and one past 2^52, where the loop's bounds
    exclude words that comp_of would resolve."""
    comps = [(0x8000, 0x18000)] if rng.random() < 0.3 else []
    pos = rng.randrange(1 << 20, 1 << 44)
    for _ in range(rng.randrange(0, 40)):
        pos += rng.choice([0, 0, rng.randrange(1, 1 << 16)])
        z = rng.choice([0, rng.randrange(1, 1 << 16)])
        comps.append((pos, pos + z))
        pos += z
    if rng.random() < 0.3:
        comps.append(((1 << 52) - 0x8000, (1 << 52) + 0x8000))
    return comps


def words(rng, comps, n):
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            v = rng.randrange(0, 0x10000)
        elif r < 0.35 and comps:
            s, e = rng.choice(comps)
            v = rng.randrange(s, e + 1)
        elif r < 0.45 and comps:
            v = rng.choice(rng.choice(comps))  # a component's first byte or its end
        elif r < 0.55:
            v = rng.randrange(1 << 52, 1 << 64)
        elif r < 0.75 and out:
            v = rng.choice(out)
        else:
            v = rng.randrange(0x10000, 1 << 52)
        out.append(v)
    return out


def test_ptr_words_matches_the_word_loop():
    rng = random.Random(185)
    for trial in range(400):
        comps = layout(rng)
        ws = words(rng, comps, rng.randrange(0, 700))
        data = b"".join(struct.pack("<Q", v) for v in ws) + bytes(rng.choice([0, 3]))
        lo = np.array([s for s, _ in comps], dtype=np.uint64)
        hi = np.array([e for _, e in comps], dtype=np.uint64)
        assert _ptr_words(data, lo, hi) == loop_words(data, comps), trial


def test_ptr_words_edges():
    lo = np.array([0x20000, 0x30000, 0x30000, 0x50000], dtype=np.uint64)
    hi = np.array([0x30000, 0x30000, 0x40000, 0x60000], dtype=np.uint64)
    comps = list(zip(lo.tolist(), hi.tolist()))
    data = struct.pack(
        "<7Q", 0x40000, 0x2FFFF, 0x1FFFF, 0xFFFF, 1 << 52, 0x40000, 0x50000
    ) + bytes(5)
    want = [0x40000, 0x2FFFF, 0x50000]
    assert _ptr_words(data, lo, hi) == loop_words(data, comps) == want
    empty = np.zeros(0, dtype=np.uint64)
    assert _ptr_words(data, empty, empty) == loop_words(data, []) == []
    assert _ptr_words(b"", lo, hi) == []
