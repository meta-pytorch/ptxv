# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch order from recorded streams and stream order calls (python/ptxv/_order.py,
ASSUMPTIONS.md E-LAUNCH-ORDER): only order CUDA enforces becomes an edge."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_order", Path(__file__).resolve().parents[1] / "ptxv" / "_order.py"
)
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)


# Records of one host thread (tid 1) unless a test says otherwise.
def L(name, sid, pdl=0, **kw):
    return {
        **dict(
            name=name, stream=sid, stream_id=sid, dev=0, pdl=pdl, capturing=0, tid=1
        ),
        **kw,
    }


def rec(ev, sid, **kw):
    return {
        **dict(
            ev="record", event=ev, stream=sid, stream_id=sid, dev=0, capturing=0, tid=1
        ),
        **kw,
    }


def wait(ev, sid, **kw):
    return {
        **dict(
            ev="wait", event=ev, stream=sid, stream_id=sid, dev=0, capturing=0, tid=1
        ),
        **kw,
    }


def destroy(ev, kind="destroy", **kw):
    return {**dict(ev=kind, event=ev, dev=0, tid=1), **kw}


def test_one_stream_chain():
    lo = O.launch_order([L("a", 1), L("b", 1), L("c", 1, pdl=1)])
    assert lo.recorded
    assert lo.after == [None, None, None] and lo.pdl == [False, False, True]
    assert lo.stream == [0, 0, 0] and lo.waits == [[], [], []]
    assert O.toml_top(lo).startswith('launch_order = "recorded"')


def test_two_streams_without_event_are_concurrent():
    lo = O.launch_order([L("a", 1), L("b", 2)])
    assert lo.recorded and lo.stream == [0, 1]
    assert lo.after == [None, -1] and lo.preds[1] == {}
    assert any("no recorded order" in n for n in lo.notes)


def test_event_orders_streams():
    lo = O.launch_order([L("a", 1), rec(7, 1), wait(7, 2), L("b", 2)])
    assert lo.after == [None, 0] and not lo.pdl[1] and lo.preds[1] == {0: "full"}
    assert any("ordered by an event" in n for n in lo.notes)


def test_event_recorded_before_launch_orders_nothing():
    lo = O.launch_order([rec(7, 1), L("a", 1), wait(7, 2), L("b", 2)])
    assert lo.after == [None, -1]


def test_destroyed_event_handle_orders_nothing():
    lo = O.launch_order(
        [L("a", 1), rec(7, 1), {"ev": "destroy", "event": 7}, wait(7, 2), L("b", 2)]
    )
    assert lo.after == [None, -1]


def test_event_handle_ends_at_destroy_and_ipc_open():
    # The driver hands a destroyed event's handle to the next new event: a wait on it before
    # it is recorded again orders nothing; once recorded, it is the new event.
    lo = O.launch_order([L("a", 1), rec(7, 1), destroy(7), wait(7, 2), L("b", 2)])
    assert lo.recorded and lo.after == [None, -1] and lo.preds[1] == {}
    lo = O.launch_order(
        [L("a", 1), rec(7, 1), destroy(7), L("c", 3), rec(7, 3), wait(7, 2), L("b", 2)]
    )
    assert lo.preds[2] == {1: "full"}
    # An event shared with another process (opened from or exported to an IPC handle): the
    # other process can record it, so it orders nothing until destroyed.
    for kind in ("ipc_open", "ipc_get"):
        lo = O.launch_order(
            [L("a", 1), rec(7, 1), destroy(7, kind), wait(7, 2), L("b", 2)]
        )
        assert lo.recorded and lo.after == [None, -1]
        lo = O.launch_order(
            [destroy(7, kind), L("a", 1), rec(7, 1), wait(7, 2), L("b", 2)]
        )
        assert lo.after == [None, -1]
    lo = O.launch_order(
        [destroy(7, "ipc_get"), destroy(7), L("a", 1), rec(7, 1), wait(7, 2), L("b", 2)]
    )
    assert lo.after == [None, 0]


def test_graph_launch_ends_event_records():
    # A graph's event record nodes record events when it runs.
    g = dict(
        ev="graph_launch", graph=11, stream=3, stream_id=3, dev=0, capturing=0, tid=1
    )
    lo = O.launch_order([L("a", 1), rec(7, 1), g, wait(7, 2), L("b", 2)])
    assert lo.recorded and lo.after == [None, -1]


def test_join_uses_waits():
    lo = O.launch_order([L("a", 1), L("b", 2), rec(5, 2), wait(5, 1), L("c", 1)])
    # c follows a on its stream and waits for b; b is the previous launch on the GPU, so
    # c names a.
    assert lo.after == [None, -1, 0] and lo.waits[2] == [1]
    lo = O.launch_order([L("b", 2), L("a", 1), rec(5, 2), wait(5, 1), L("c", 1)])
    assert lo.after == [None, -1, None] and lo.waits[2] == [0]


def test_transitive_reduction():
    # b waits for a; c (stream 1, after a) waits for b: a is implied by b.
    lo = O.launch_order(
        [L("a", 1), rec(5, 1), wait(5, 2), L("b", 2), rec(6, 2), wait(6, 1), L("c", 1)]
    )
    assert lo.preds[2] == {1: "full"} and lo.after[2] == 1 and lo.waits[2] == []


def test_programmatic_event():
    pe = {"event": 9, "trigger_at_block_start": 0}
    lo = O.launch_order([L("a", 1, prog_event=pe), wait(9, 2), L("b", 2)])
    assert lo.after == [None, 0] and lo.pdl == [False, True]
    pe = {"event": 9, "trigger_at_block_start": 1}
    lo = O.launch_order([L("a", 1, prog_event=pe), wait(9, 2), L("b", 2)])
    assert lo.after == [None, -1] and not lo.pdl[1]
    lo = O.launch_order([L("a", 1, completion_event=9), wait(9, 2), L("b", 2)])
    assert lo.after == [None, -1]


def test_external_event_flags_order_nothing():
    lo = O.launch_order([L("a", 1), rec(7, 1, flags=1), wait(7, 2), L("b", 2)])
    assert lo.after == [None, -1]
    lo = O.launch_order([L("a", 1), rec(7, 1), wait(7, 2, flags=1), L("b", 2)])
    assert lo.after == [None, -1]


def test_programmatic_launch_keeps_its_predecessor_in_the_event():
    # b may finish before a (programmatic): an event after b covers both.
    lo = O.launch_order([L("a", 1), L("b", 1, pdl=1), rec(7, 1), wait(7, 2), L("c", 2)])
    assert lo.preds[2] == {0: "full", 1: "full"}


def test_projection_through_dropped_launches():
    recs = [L("a", 1), L("x", 1), L("b", 1)]
    assert O.launch_order(recs, [0, 2]).preds[1] == {0: "full"}
    recs = [L("a", 1), L("x", 1), L("b", 1, pdl=1)]
    lo = O.launch_order(recs, [0, 2])
    assert lo.preds[1] == {0: "pdl"} and lo.pdl[1] and lo.after[1] is None
    recs = [L("a", 1), L("x", 1, pdl=1), L("b", 1)]
    assert O.launch_order(recs, [0, 2]).after[1] == -1


def test_graph_capture_is_its_own_namespace():
    cap = dict(capturing=1, capture_id=3)
    recs = [
        L("e0", 1),
        dict(L("g0", 1), **cap),
        rec(5, 1, **cap),
        wait(5, 2),  # stream 2 joins the capture
        dict(L("g1", 2), **cap),
        rec(6, 2, **cap),
        wait(6, 1, **cap),
        dict(L("g2", 1), **cap),
        dict(
            ev="graph_launch",
            graph=11,
            stream=1,
            stream_id=1,
            dev=0,
            capturing=0,
            tid=1,
        ),
        L("e1", 1),
    ]
    lo = O.launch_order(recs)
    assert lo.recorded
    # g0 has no edge to e0; g1 after g0 (fork); g2 after g0 (stream) and g1 (join); e1 after e0.
    assert lo.preds[1] == {} and lo.preds[2] == {1: "full"}
    assert lo.preds[3] == {2: "full"}
    assert lo.preds[4] == {0: "full"}


def test_capture_dependencies_rewritten():
    # cuStreamUpdateCaptureDependencies: the stream's next launch in the capture depends on
    # the nodes the call set, not on the stream's earlier launches.
    cap = dict(capturing=1, capture_id=3)
    deps = dict(ev="capture_deps", nodes=0, flags=1, stream=1, dev=0, tid=1, **cap)
    lo = O.launch_order([dict(L("g0", 1), **cap), deps, dict(L("g1", 1), **cap)])
    assert lo.recorded and lo.preds[1] == {} and lo.after == [None, -1]


def test_per_thread_streams_in_a_capture_are_the_threads_own():
    # Handle 0 through a _ptsz entry point (or CU_STREAM_PER_THREAD) names each thread's own
    # stream: two threads' launches in one capture are not stream-ordered.
    cap = dict(capturing=1, capture_id=3, stream_id=None, ptsz=1, stream=0)
    a = dict(L("a", 0), t0=0, t1=1, **cap)
    b = dict(L("b", 0), tid=2, t0=2, t1=3, **cap)
    lo = O.launch_order([a, b])
    assert lo.recorded and lo.preds[1] == {} and lo.stream == [0, 1]
    assert O.launch_order([a, dict(b, tid=1)]).preds[1] == {0: "full"}


def test_graph_capture_without_stream_ids():
    # As recorded: a capturing stream has no id (the query would invalidate the capture).
    def C(rec_):
        r = dict(rec_, capturing=1, capture_id=3)
        r.pop("stream_id")
        return r

    recs = [
        C(L("g0", 1)),
        C(rec(5, 1)),
        wait(5, 2),  # stream 2 (id known: not capturing yet) joins the capture
        C(L("g1", 2)),
        C(rec(6, 2)),
        C(wait(6, 1)),
        C(L("g2", 1)),
    ]
    lo = O.launch_order(recs)
    assert lo.recorded
    assert lo.preds[1] == {0: "full"} and lo.preds[2] == {1: "full"}
    assert lo.after == [None, 0, 1] and lo.stream == [0, 1, 0]


def test_launches_on_two_gpus_are_unrecorded():
    # One process, two GPUs: the case has no per-launch device.
    lo = O.launch_order([L("a", 1, dev=0), L("b", 2, dev=1)])
    assert not lo.recorded and "several GPUs" in lo.why
    # tools/capture_dist.py: one GPU per process.
    assert O.launch_order([L("a", 1), L("b", 2)], single_device=True).recorded


def test_several_threads():
    a, b = L("a", 1, t0=0, t1=10), L("b", 2, tid=2, t0=20, t1=30)
    assert O.launch_order([a, b]).recorded
    # b's call was made before a's returned: the driver may have taken b first.
    lo = O.launch_order([dict(a, t1=25), b])
    assert not lo.recorded and "overlap" in lo.why
    lo = O.launch_order(
        [a, rec(7, 1, t0=11, t1=21), wait(7, 2, tid=2, t0=22, t1=23), b]
    )
    assert not lo.recorded and "overlap" in lo.why
    # Several threads without times, and records without a thread (an older library).
    assert not O.launch_order([L("a", 1), L("b", 2, tid=2)]).recorded
    lo = O.launch_order([dict(name="a", stream=1, stream_id=1), L("b", 1)])
    assert not lo.recorded and "thread id" in lo.why


def test_unrecorded_inputs():
    lo = O.launch_order([dict(name="a", stream=0), dict(name="b", stream=0)])
    assert not lo.recorded and "stream id" in lo.why
    assert O.toml_top(lo).endswith('launch_order = "unknown"\n')
    assert O.toml_launch(lo, 1, True) == "pdl = true\n"
    lo = O.launch_order([L("a", 1), L("b", 1, unrecorded_before=2)])
    assert not lo.recorded
    lo = O.launch_order([L("a", 1), L("b", 1)], [1, 0])
    assert not lo.recorded


def test_place_maps_processes_into_the_case():
    a = O.launch_order([L("a", 1), L("b", 2), rec(5, 2), wait(5, 1), L("c", 1)])
    b = O.launch_order([L("x", 1), L("y", 1)])
    lo = O.place([(a, [0, 2, 4]), (b, [1, 3])], 5)
    assert lo.recorded
    assert lo.after == [None, None, -1, None, 0] and lo.waits[4] == [2]
    assert O.toml_launch(lo, 4, False) == "stream = 0\nafter = 0\nwaits = [2]\n"
