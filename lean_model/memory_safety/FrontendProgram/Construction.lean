/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar
import FrontendProgram.SourceDimensions

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

abbrev Manifest := ptxv_case.Manifest
abbrev Buffer := ptxv_case.Buffer
abbrev Program := ptxv_memory_write.execution_core.Program
abbrev Allocation := ptxv_memory_write.execution_core.AllocationBytes
abbrev Layout := ptxv_memory_write.value_flow.AllocationLayout

def SupportedManifest (manifest : Manifest) : Prop :=
  manifest.cubin=none ∧ manifest.executed_cubin=none ∧ manifest.dynamic_smem.val=0 ∧
  manifest.cluster=none ∧ manifest.preferred_cluster=none ∧ manifest.replicas.val=[] ∧
  manifest.spec=none ∧ manifest.mma_model=none ∧ manifest.exit_fence=none ∧ manifest.poll_acquire=false ∧
  manifest.assume_refuted.val=[] ∧ manifest.smem_cumulative=false ∧ manifest.hw_l1=false ∧
  manifest.l1_coresident=false ∧ manifest.l1_havoc=false ∧ manifest.residency=none ∧
  manifest.cooperative=false ∧ manifest.tmem_cotenant=false ∧ manifest.read_done=false ∧
  manifest.remote_fifo=false ∧ manifest.domains.val=[] ∧ manifest.nic_qps.val=[] ∧ manifest.nic_mrs.val=[] ∧
  manifest.multicast.val=[] ∧ manifest.mc_alias=none ∧ manifest.views.val=[] ∧ manifest.spaces.val=[] ∧
  manifest.alias_xspace=none ∧ manifest.params.val=[] ∧ manifest.launches.val=[] ∧ manifest.permuted.val=[]

def BufferExtent (buffer : Buffer) (bytes : Nat) : Prop :=
  ∃ width, (buffer.dtype,width) ∈ [("u8",1),("i8",1),("u16",2),("i16",2),("u32",4),("i32",4),("u64",8),("i64",8)] ∧
    bytes=width*(buffer.shape.val.map (fun value => value.val)).prod

def BufferSupported (buffer : Buffer) : Prop :=
  buffer.device.val=0 ∧ buffer.symbol≠none ∧ buffer.cubin=none ∧
  (buffer.scratch=true → buffer.init="undef") ∧ buffer.file=none ∧ buffer.expect=none ∧
  buffer.expect_offset.val=0 ∧ buffer.expect_ignore.val=[] ∧ buffer.values.val=[] ∧
  buffer.ptrs.val=[] ∧ buffer.invariant=none ∧ (buffer.init="zeros" ∨ buffer.init="undef")

def BacksObject (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (object : Nat) (buffer : Buffer) : Prop :=
  ∃ global symbol, parsed.scope.globals.val[object]?=some global ∧ buffer.symbol=some symbol ∧
    nameBytes source global.name=utf8 symbol

def BufferInventory (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest) : Prop :=
  (∀ buffer ∈ manifest.buffers.val, BufferSupported buffer ∧
    ∃ object global, parsed.scope.globals.val[object]?=some global ∧ BacksObject source parsed object buffer ∧
      BufferExtent buffer global.bytes.val.length) ∧
  manifest.buffers.val.Pairwise (fun left right => left.name≠right.name) ∧
  (∀ (i j object : Nat) (left right : Buffer), manifest.buffers.val[i]?=some left → manifest.buffers.val[j]?=some right →
    BacksObject source parsed object left → BacksObject source parsed object right → i=j)

def InitialAllocation (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) (allocation : Allocation) : Prop :=
  ∃ global, parsed.scope.globals.val[object]?=some global ∧ allocation.live=true ∧
    ((¬∃ buffer ∈ manifest.buffers.val, BacksObject source parsed object buffer) ∧
      allocation.bytes.val=global.bytes.val ∧ allocation.defined.val=List.replicate global.bytes.val.length true ∨
    ∃ buffer ∈ manifest.buffers.val, BacksObject source parsed object buffer ∧
      allocation.bytes.val=List.replicate global.bytes.val.length 0#u8 ∧
      allocation.defined.val=List.replicate global.bytes.val.length (buffer.init=="zeros"))

def LayoutsMeaning (globals : List Global) (layouts : List Layout) : Prop :=
  layouts.length=globals.length ∧
  (∀ (object : Nat) (global : Global), globals[object]?=some global → ∃ (layout : Layout), layouts[object]?=some layout ∧
    layout.bytes.val=global.bytes.val.length ∧ layout.alignment=global.alignment ∧
    global.alignment.val ∣ layout.base.val ∧ layout.base.val+layout.bytes.val≤2^64) ∧
  layouts.Pairwise (fun (left right : Layout) => left.base.val+left.bytes.val≤right.base.val ∨ right.base.val+right.bytes.val≤left.base.val)

def InitialRegisters (virtuals predicates : Nat) (registers : ptxv_memory_write.value_flow.Registers) : Prop :=
  registers.raw.regs.val=List.replicate virtuals 0#u64 ∧
  registers.pointers.val=List.replicate virtuals none ∧
  registers.raw.written.val=List.replicate ((virtuals+63)/64) 0#u64 ∧
  registers.raw.preds.val=List.replicate ((predicates+63)/64) 0#u64 ∧
  registers.raw.pdef.val=List.replicate ((predicates+63)/64) 0#u64 ∧ registers.raw.vund.val=[]

def InitialThread (manifest : Manifest) (virtuals predicates index : Nat)
    (thread : ptxv_memory_write.execution_core.ThreadState) : Prop :=
  let grid := dimensionsModel manifest.grid
  let block := dimensionsModel manifest.block
  thread.pc.val=0 ∧ thread.running=true ∧ InitialRegisters virtuals predicates thread.registers ∧
  thread.ids.ntid=manifest.block ∧ thread.ids.nctaid=manifest.grid ∧
  dimensionsModel thread.ids.ctaid=PtxFrontend.coordinates grid (index/block.size) ∧
  dimensionsModel thread.ids.tid=PtxFrontend.coordinates block (index%block.size) ∧
  thread.ids.lane.val=index%block.size%32

def ThreadsMeaning (manifest : Manifest) (virtuals predicates : Nat)
    (threads : List ptxv_memory_write.execution_core.ThreadState) : Prop :=
  let grid := dimensionsModel manifest.grid
  let block := dimensionsModel manifest.block
  grid.positive ∧ block.positive ∧ threads.length=grid.size*block.size ∧
  ∀ index thread, threads[index]?=some thread → InitialThread manifest virtuals predicates index thread

def OutputsMeaning (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (outputs : List Usize) : Prop :=
  outputs.length=manifest.outputs.val.length ∧
  ∀ (position : Nat) (object : Usize) (label : String), outputs[position]?=some object → manifest.outputs.val[position]?=some label →
    ∃ buffer ∈ manifest.buffers.val, buffer.name=label ∧ BacksObject source parsed object.val buffer

def ConstructionMeaning (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (program : Program) : Prop :=
  utf8 manifest.entry=entry ∧ SupportedManifest manifest ∧ BufferInventory source parsed manifest ∧
  program.code=parsed.code ∧ LayoutsMeaning parsed.scope.globals.val program.layouts.val ∧
  program.initial.allocations.val.length=parsed.scope.globals.val.length ∧
  (∀ (object : Nat) (allocation : Allocation), program.initial.allocations.val[object]?=some allocation → InitialAllocation source parsed manifest object allocation) ∧
  ThreadsMeaning manifest parsed.scope.virtual_count.val parsed.scope.predicate_count.val program.initial.threads.val ∧
  OutputsMeaning source parsed manifest program.outputs.val

def SourceProgram (source entry : List U8) (manifest : Manifest) (program : Program) : Prop :=
  ∃ parsed, ModuleMeaning source entry parsed ∧ ConstructionMeaning source entry manifest parsed program

end PtxFrontendProgram
