# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The launch order a capture recorded (ASSUMPTIONS.md E-LAUNCH-ORDER, docs/capture.md).

`launch_order(records, kept)` turns one process's recorded launches and stream order calls, in
the order the process issued them, into the predecessor relations of a case: for each kept
launch its stream, `after` (one predecessor; -1: none), `pdl` (that edge is programmatic) and
`waits` (further predecessors it starts after they completed). Only order CUDA enforces
becomes an edge:

- the previous launch on the same stream: a full edge, or a programmatic one when the launch
  carries programmatic stream serialization ("overlap with the previous kernel in the
  stream", CUDA 13.0 cuda.h); an event wait stays a full edge;
- an event recorded on stream S1 and waited for on stream S2: S2's next launch starts after
  every launch S1 had been given completed (the last one, and the ones a programmatic launch
  did not wait for). An event recorded through a launch's ProgrammaticEvent attribute gives a
  programmatic edge to the waiting stream's next launch; with triggerAtBlockStart, or as a
  LaunchCompletionEvent ("blocks of B may begin before all blocks of A have begun"), none;
- stream capture: the same rules among the launches of one capture sequence (the graph's
  edges; a stream that waits for an event recorded in a capture joins it; after
  cuStreamUpdateCaptureDependencies the stream's next launch has no edge from what it was
  given before), and no edge between a captured launch and any other (the graph runs when it
  is launched, maybe more than once);
- a graph launch is opaque work on its stream: the launches around it keep their stream
  order through it.

Order the records do not show (host synchronization, the legacy default stream's implicit
synchronization, stream memory operations, events with the external flag, events recorded
before the records start) gives no edge. That only adds behaviours, with one exception at the
strict level: a programmatic edge from a ProgrammaticEvent wait is carried to the waiting
stream's next launch even when an unrecorded stream operation (a memset, a copy, a host
function) sits between them, an edge CUDA does not promise (GB300 runs it no earlier than the
primary's completion). An event's record ends at cuEventDestroy (the driver hands the handle
to a new event, and a wait on a never-recorded event orders nothing) and at a graph launch (a
graph's event record nodes record events the records do not show); an event shared with
another process (cuIpcOpenEventHandle, cuIpcGetEventHandle) orders nothing until destroyed,
since the other process can record it. Launches that are not kept (filtered out) still order
the kept ones through them: a kept launch's predecessor through a dropped one is kept when
the path is full edges, or full edges then a final programmatic one.

`recorded` (the case may say `launch_order = "recorded"`) needs every launch record to name
its stream by id (cuStreamGetId; a handle may be reused, and stream 0 means the legacy or the
calling thread's stream depending on the entry point), or else to be issued into a graph
capture with a capture id (its stream is then named by handle within the capture); no launch
left out of the records between kept ones ("unrecorded_before", injection mode); the kept
launches in issue order and on one GPU (a case written by one process has no per-launch
device); and every record naming its host thread ("tid"), with the calls of different threads
apart in time: each one's "t0" (just before the call) after every earlier record's "t1" (after
the call returned) of another thread. The records of one thread are in the order the driver
took its calls; those of two threads only when one call returned before the other was made.

Records (dicts, probes/launchcap and python/ptxv/_native/ptxvcap.c JSON lines), written only
for calls that succeeded: a launch has "name", "stream" (handle), "stream_id", "tid", "t0",
"t1", optional "dev", "pdl", "capturing" (1, 0, -1 unknown), "capture_id", "prog_event"
({"event", "trigger_at_block_start"}), "completion_event", "mem_sync_domain",
"unrecorded_before"; a stream order call has "ev" = "record" | "wait" | "graph_launch" |
"capture_deps" with "event" (or "graph", "nodes"), "flags", "stream", "stream_id", "dev",
"capturing", "capture_id", "tid", "t0", "t1", or "ev" = "destroy" | "ipc_open" | "ipc_get"
with "event", "tid", "t0", "t1". Other records are ignored.
"""

# No `from __future__ import annotations`: tools load this file by path (no package), where
# dataclasses cannot resolve string annotations.
from dataclasses import dataclass, field

FULL, PDL = (
    2,
    1,
)  # edge kinds: completion (with its memory); programmatic (start after triggers)
EXTERNAL = 1  # CU_EVENT_RECORD_EXTERNAL / CU_EVENT_WAIT_EXTERNAL
ORDER_CALLS = (
    "record",
    "wait",
    "graph_launch",
    "capture_deps",
    "destroy",
    "ipc_open",
    "ipc_get",
)


@dataclass
class LaunchOrder:
    """Per kept launch (in `kept` order): `stream` (index per device by first use), `after`
    (index into `kept`; None: implicit, the previous kept launch on the device, which is on
    the same stream; -1: none), `pdl`, `waits` (indices into `kept`), `preds` ({kept index:
    "full" | "pdl"}, every recorded predecessor after transitive reduction), `dropped`
    (predecessors no field holds: a second programmatic one) and `mem_sync_domain`.
    `recorded`: the order is complete enough to rely on (else `why`)."""

    stream: list = field(default_factory=list)
    after: list = field(default_factory=list)
    pdl: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    mem_sync_domain: list = field(default_factory=list)
    device: list = field(default_factory=list)
    recorded: bool = True
    why: str = ""
    notes: list = field(default_factory=list)


def merge_by_oseq(launches: list, calls: list) -> list:
    """ptxvcap's launches.jsonl and events.jsonl records in issue order ("oseq"). Without
    "oseq" on every record (an older library) the launches alone, which then lack stream ids
    and give an unrecorded order."""
    if not all("oseq" in r for r in launches) or not all("oseq" in r for r in calls):
        return list(launches)
    return sorted(list(launches) + list(calls), key=lambda r: r["oseq"])


def _dev(r: dict) -> int:
    d = r.get("dev", 0)
    return 0 if d is None or d < 0 else d


def _skey(r: dict, ns=None):
    """Stream identity: outside a capture (device, stream id), else (device, handle, flavour).
    A capturing stream has no id (cuStreamGetId would invalidate the capture); within one
    capture sequence its handle names it (a stream is not destroyed while capturing), and the
    thread too for a per-thread default stream (handle 0 through a `_ptsz` entry point, or
    CU_STREAM_PER_THREAD), which is each thread's own."""
    sid = r.get("stream_id")
    if sid is not None and ns is None:
        return (_dev(r), ("id", sid))
    h, ptsz = r.get("stream", 0), r.get("ptsz", 0)
    own = (h == 0 and ptsz) or h == 2  # CU_STREAM_PER_THREAD
    return (_dev(r), ("h", h, ptsz, r.get("tid") if own else None))


def launch_order(  # noqa: C901 (one pass over the records, then projection and encoding)
    records: list, kept: list | None = None, *, single_device: bool = False
) -> LaunchOrder:
    """`records`: one process's records in issue order. `kept`: indices into `records` of the
    case's launches, in case order (default: every launch record). `single_device`: the case
    puts every kept launch on one GPU (its device then is 0 for the encoding)."""
    launches = [i for i, r in enumerate(records) if "name" in r]
    if kept is None:
        kept = launches
    kept_pos = {ri: k for k, ri in enumerate(kept)}
    out = LaunchOrder()

    def unrecorded(why: str) -> None:
        if out.recorded:
            out.recorded, out.why = False, why

    if any(
        records[i].get("stream_id") is None
        and not (records[i].get("capturing") == 1 and records[i].get("capture_id"))
        for i in launches
    ):
        unrecorded("launches without a stream id (an older capture library)")
    if any(records[i].get("unrecorded_before") for i in launches[1:]):
        unrecorded(
            "launches between the recorded ones were not written (injection mode filter)"
        )
    if any(b <= a for a, b in zip(kept, kept[1:])):
        unrecorded("the case's launches are not in issue order")
    if not single_device and len({_dev(records[i]) for i in kept}) > 1:
        unrecorded(
            "launches on several GPUs in one process (the case has no per-launch device)"
        )
    calls = [r for r in records if "name" in r or r.get("ev") in ORDER_CALLS]
    if any(r.get("tid") is None for r in calls):
        unrecorded("records without a thread id (an older capture library)")
    elif len({r["tid"] for r in calls}) > 1:
        last_t1: dict = {}  # thread -> latest "t1" of its records so far
        for r in calls:
            t0, t1 = r.get("t0"), r.get("t1")
            if (
                t0 is None
                or t1 is None
                or any(v >= t0 for u, v in last_t1.items() if u != r["tid"])
            ):
                unrecorded(
                    "calls of several host threads overlap in time (the driver may have taken them in another order)"
                )
                break
            last_t1[r["tid"]] = max(last_t1.get(r["tid"], t1), t1)

    def ns_of(i: int, r: dict):
        c = r.get("capturing", 0)
        if c == 1:
            return ("capture", r.get("capture_id", 0))
        if c:  # unknown: a namespace of its own, no edge
            return ("unknown", i)
        return None

    direct: dict[int, dict[int, int]] = (
        {}
    )  # launch record -> {predecessor record: kind}
    last: dict = {}  # (namespace, stream) -> last launch record
    tail: dict = (
        {}
    )  # (namespace, stream) -> launches whose completion "all prior work" needs
    pend: dict = (
        {}
    )  # (namespace, stream) -> {record: kind} from event waits not yet consumed
    events: dict = {}  # event -> (namespace, {record: kind}) of its latest record
    shared: set = set()  # events another process can record (IPC)
    for i, r in enumerate(records):
        ns = ns_of(i, r)
        key = (ns, _skey(r, ns))
        if "name" in r:
            preds: dict[int, int] = {}
            if key in last:
                preds[last[key]] = PDL if r.get("pdl") else FULL
            for p, kd in pend.pop(key, {}).items():
                preds[p] = max(preds.get(p, 0), kd)
            direct[i] = preds
            # A programmatic launch may complete before its stream predecessor: "all prior work
            # on the stream" then still needs the earlier ones.
            tail[key] = (tail.get(key, set()) if r.get("pdl") else set()) | {i}
            last[key] = i
            pe = r.get("prog_event")
            if pe and pe.get("event"):
                events[pe["event"]] = (
                    ns,
                    {} if pe.get("trigger_at_block_start") else {i: PDL},
                )
            if r.get("completion_event"):
                events[r["completion_event"]] = (ns, {})
            continue
        ev = r.get("ev")
        if ev == "record":
            fr: dict[int, int] = {}
            if not (r.get("flags", 0) & EXTERNAL):
                fr = {p: kd for p, kd in pend.get(key, {}).items() if kd == FULL}
                for t in tail.get(key, ()):
                    fr[t] = FULL
            if r.get("event") in shared:
                events.pop(r.get("event"), None)
            else:
                events[r.get("event")] = (ns, fr)
        elif ev == "destroy":
            # cuEventCreate hands destroyed handles out again: a new event.
            events.pop(r.get("event"), None)
            shared.discard(r.get("event"))
        elif ev in ("ipc_open", "ipc_get"):
            events.pop(r.get("event"), None)
            shared.add(r.get("event"))
        elif ev == "graph_launch":
            # A graph's event record nodes record events when it runs.
            events.clear()
        elif ev == "capture_deps":
            # The capturing stream's next node depends on what the call set, not on what the
            # stream was given before.
            last.pop(key, None)
            tail.pop(key, None)
            pend.pop(key, None)
        elif ev == "wait":
            got = events.get(r.get("event"))
            if got is None or r.get("flags", 0) & EXTERNAL:
                continue
            ens, fr = got
            if ens != ns:
                if ns is None and ens is not None and ens[0] == "capture":
                    # The waiting stream joins the capture: its next launches are captured.
                    key = (ens, _skey(r, ens))
                else:
                    continue
            p = pend.setdefault(key, {})
            for q, kd in fr.items():
                p[q] = max(p.get(q, 0), kd)
        # A graph launch is also opaque work on its stream: its neighbours keep their stream
        # order through it.

    # Projection onto the kept launches.
    proj: dict[int, dict[int, int]] = {}

    def projected(i: int) -> dict[int, int]:
        if i in proj:
            return proj[i]
        res: dict[int, int] = {}
        for p, kd in direct.get(i, {}).items():
            if p in kept_pos:
                res[p] = max(res.get(p, 0), kd)
                continue
            for a, ka in projected(p).items():
                if ka == FULL:
                    res[a] = max(res.get(a, 0), kd)
        proj[i] = res
        return res

    def edev(ri: int) -> int:
        return 0 if single_device else _dev(records[ri])

    hb_anc: dict[int, set[int]] = {}  # all-full ancestors among kept launches
    preds_of: dict[int, dict[int, int]] = {}
    for ri in kept:
        pr = {
            p: kd
            for p, kd in projected(ri).items()
            if edev(p) == edev(ri) and kept_pos[p] < kept_pos[ri]
        }
        red = {
            p: kd
            for p, kd in pr.items()
            if not any(kq == FULL and q != p and p in hb_anc[q] for q, kq in pr.items())
        }
        preds_of[ri] = red
        anc: set[int] = set()
        for p, kd in red.items():
            if kd == FULL:
                anc |= {p} | hb_anc[p]
        hb_anc[ri] = anc

    snum: dict = {}
    prev_on_dev: dict = (
        {}
    )  # device -> (record, stream index) of the previous kept launch
    prev_on_stream: dict = {}  # (device, stream index) -> previous kept launch on it
    for k, ri in enumerate(kept):
        r = records[ri]
        dev = edev(ri)
        skey = (dev, ns_of(ri, r), _skey(r, ns_of(ri, r)))
        if skey not in snum:
            snum[skey] = sum(1 for x in snum if x[0] == dev)
        s = snum[skey]
        pr = preds_of[ri]
        pd = sorted((p for p, kd in pr.items() if kd == PDL), key=kept_pos.get)
        fu = sorted((p for p, kd in pr.items() if kd == FULL), key=kept_pos.get)
        prev = prev_on_dev.get(dev)
        implicit = prev is not None and prev[1] == s
        dropped = []
        if pd:
            a, pdl = pd[-1], True
            dropped = [kept_pos[p] for p in pd[:-1]]
            ws = fu
        elif fu:
            # `after`: the previous launch on the stream when it is one (left implicit when it
            # is also the previous one on the GPU), else the latest.
            ps = prev_on_stream.get((dev, s))
            a = ps if ps in fu else fu[-1]
            pdl = False
            ws = [p for p in fu if p != a]
        else:
            a, pdl, ws = None, False, []
        if a is None:
            after = None if prev is None else -1
        elif implicit and a == prev[0]:
            after = None
        else:
            after = kept_pos[a]
        if prev is not None and prev[1] != s:
            ordered = prev[0] in pr or prev[0] in hb_anc[ri]
            out.notes.append(
                f"launch {k} ({r['name'][:50]}) on stream {s} after launch {kept_pos[prev[0]]} on stream {prev[1]}: "
                + (
                    "ordered by an event"
                    if ordered
                    else "no recorded order between them (they may overlap)"
                )
            )
        prev_on_dev[dev] = (ri, s)
        prev_on_stream[(dev, s)] = ri
        out.stream.append(s)
        out.after.append(after)
        out.pdl.append(pdl)
        out.waits.append([kept_pos[p] for p in ws])
        out.preds.append(
            {
                kept_pos[p]: ("full" if kd == FULL else "pdl")
                for p, kd in sorted(pr.items(), key=lambda x: kept_pos[x[0]])
            }
        )
        out.dropped.append(dropped)
        out.mem_sync_domain.append(int(r.get("mem_sync_domain") or 0))
        out.device.append(dev)
        for d in dropped:
            out.notes.append(
                f"launch {k}: a second programmatic predecessor (launch {d}) left out (no edge)"
            )
        if r.get("capturing"):
            out.notes.append(
                f"launch {k} ({r['name'][:50]}) was issued into a graph capture: ordered only among the launches of its capture"
            )
    return out


def place(parts: list, n: int) -> LaunchOrder:
    """One case's order from several processes' (tools/capture_dist.py: one per GPU):
    `parts` = [(LaunchOrder, [case index of each of its kept launches])], every process on a
    GPU of its own and its launches in its own order in the case."""
    out = LaunchOrder()
    out.stream, out.after, out.pdl = [0] * n, [None] * n, [False] * n
    out.waits, out.preds, out.dropped = (
        [[] for _ in range(n)],
        [{} for _ in range(n)],
        [[] for _ in range(n)],
    )
    out.mem_sync_domain, out.device = [0] * n, [0] * n
    for lo, idx in parts:
        if not lo.recorded:
            out.recorded = False
            out.why = out.why or lo.why
        for k, g in enumerate(idx):
            out.stream[g] = lo.stream[k]
            a = lo.after[k]
            out.after[g] = a if a is None or a < 0 else idx[a]
            out.pdl[g] = lo.pdl[k]
            out.waits[g] = [idx[w] for w in lo.waits[k]]
            out.preds[g] = {idx[p]: kd for p, kd in lo.preds[k].items()}
            out.dropped[g] = [idx[d] for d in lo.dropped[k]]
            out.mem_sync_domain[g] = lo.mem_sync_domain[k]
        out.notes += lo.notes
    return out


def toml_top(lo: LaunchOrder) -> str:
    """The case's top-level order lines: `launch_order`, and the first launch's `stream` and
    `mem_sync_domain`."""
    if not lo.recorded:
        return f'# launch order not recorded: {lo.why}\nlaunch_order = "unknown"\n'
    s = 'launch_order = "recorded"\n' + f"stream = {lo.stream[0]}\n"
    if lo.mem_sync_domain[0]:
        s += f"mem_sync_domain = {lo.mem_sync_domain[0]}\n"
    return s


def toml_launch(lo: LaunchOrder, k: int, raw_pdl: bool) -> str:
    """The order lines of `[[launches]]` entry k (k >= 1). Unrecorded: the launch's own PDL
    attribute against the list's previous launch, as before."""
    if not lo.recorded:
        return "pdl = true\n" if raw_pdl else ""
    s = f"stream = {lo.stream[k]}\n"
    if lo.after[k] is not None:
        s += f"after = {lo.after[k]}\n"
    if lo.pdl[k]:
        s += "pdl = true\n"
    if lo.waits[k]:
        s += f"waits = {lo.waits[k]}\n"
    if lo.mem_sync_domain[k]:
        s += f"mem_sync_domain = {lo.mem_sync_domain[k]}\n"
    return s


def summary(lo: LaunchOrder) -> dict:
    """capture.json's "launch_order"."""
    return {
        "recorded": lo.recorded,
        **({"why": lo.why} if not lo.recorded else {}),
        "launches": [
            {
                "stream": lo.stream[k],
                "after": lo.after[k],
                "pdl": lo.pdl[k],
                "waits": lo.waits[k],
                "preds": {str(p): kd for p, kd in lo.preds[k].items()},
                "mem_sync_domain": lo.mem_sync_domain[k],
            }
            for k in range(len(lo.stream))
        ],
        "notes": lo.notes,
    }
