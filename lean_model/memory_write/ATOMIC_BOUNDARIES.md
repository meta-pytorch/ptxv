# Atomic verification boundaries

The extracted atomic core proves a single selected atomic operation: its integer
result or floating arithmetic request, width, CAS outcome, returned value,
predicate, epoch decision, and memory-effect handoff. Floating addition is a
contract of the existing arithmetic subsystem. The surrounding scheduler,
happens-before clocks, cache behavior, multicast binding, and NIC handling have
their existing contracts; the core theorem does not establish their soundness.

## Floating arithmetic

The callback contract names the existing implementation's word functions. It
does not assume they match the standalone FloatLib atomic model. For example,
`Shared/Atomic.lean` specifies space-dependent f64 NaN propagation, while the
current production adapter calls the same `fp::f64_fma` path for both spaces.
The existing `probes/atom_f64` corpus includes a quiet NaN, but does not establish
the signaling-NaN and competing-payload cases needed to resolve this difference.

## Independent component ordering

The continuation exposes all lane/component choices for return-bearing f32
v2/v4 atomics. Returnless vector RED and packed bf16x2 retain whole-instruction
scheduling and cannot represent every allowed component interleaving. This affects
returned observations and observers even when final memory agrees.

Start with two components `[0, 0]`. Thread A adds `[1, 10]`; thread B adds `[2, 20]`.
The component order `A.low, B.low, B.high, A.high` gives:

| Observation | Value |
| --- | --- |
| A's returned old components | `[0, 20]` |
| B's returned old components | `[1, 0]` |
| Final memory | `[3, 30]` |

Serializing both entire operations as A then B instead returns A=`[0, 0]`,
B=`[1, 10]`. B then A returns A=`[2, 20]`, B=`[0, 0]`. Both whole-operation orders
have the same final memory as the componentwise execution. These small integers
and sums are exactly representable in f32 and bf16. The PTX element-atomicity
contract already cited by the vector handler permits independent component
orders; there is no whole-vector atomicity guarantee.

[AtomicComponentOrder.lean](MemoryWrite/AtomicComponentOrder.lean) constructs the
two component histories and proves in the Lean kernel that their combined
returns differ from either whole-operation order, despite identical finals.
This theorem concerns the component transition model; it does not prove a GPU
architecture contract or completeness of the simulator's exploration.

## Production continuation and proof boundary

`exec/atomic_components.rs` freezes the original issued split, guards, addresses,
source bits and numeric metadata before any component writes back. The persistent
continuation is hashed. `atomic_components::choices` exposes every pending
component, and `begin_component` returns its frozen operand/address and clears
exactly that choice. Every component calls the existing scalar atomic path, so
bounds, shared-target liveness, memory effects and scalar clocks are checked at
the actual access. Issued lane PCs advance together only after all components.
Later arrivals cannot join or overwrite the issued mask.

The extracted planner and prefix theorems prove arbitrary component orders,
exact-once progress, frozen sources, returned-event traces, sticky findings and
future enabledness in the component model. Native Machine, DPOR, scalar arithmetic,
clock publication and diagnostic adapters are exercised by regressions; these
are not a whole-explorer completeness proof or a new certificate soundness proof.

Issue validates and captures all active operands before writing any component.
A later-lane issue fault therefore reports and stops the issued split with no
component writes. Direct `exec_one`, used for copied-machine lookahead, retains
its legacy lane loop and may leave earlier lane writes before returning an error.
The successful-latching refinement does not equate these failed-issue prefixes.

The certificate logs the original guard and issued mask once, then each actual
scalar component under a component instruction view. C-TICKET's current forcing
identity is a whole instruction at one word. Its extracted eligibility policy
excludes **every** word touched by a vector f32 atomic, including upper components
and other lanes; C-ORDER then requires those accesses to be ordered or refuses
the certificate. Disjoint vector certificates and scalar ticket gates remain
supported. Component-keyed ticket gates are a future extension.

Returnless RED is unchanged: exposing all choices exceeded a bounded native
sample before reaching a terminal state, and no unproved fixed component order
was introduced. Packed bf16x2 still computes both halves in one four-byte access;
it needs selected-half provenance and containing-word marker preservation when
component scheduling is added.
