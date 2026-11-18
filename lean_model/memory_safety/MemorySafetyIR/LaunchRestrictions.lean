/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

structure KernelRestrictions (kernel : ptxv_ir.Kernel) : Prop where
  ptx : kernel.front = .Ptx
  params : kernel.params.val = []
  parts : kernel.parts.val = []
  findings : kernel.static_findings.val = []
  bulkReadWaits : kernel.bulk_read_waits.val = []
  bulkCommit : kernel.bulk_commit_sb.val = []
  staticImage : kernel.static_image = none
  nonempty : kernel.code.val ≠ []
  ldtmScoreboards : kernel.ldtm_sb.val = []
  ldtmWaits : kernel.ldtm_waits.val = []

structure GridRestrictions (grid : machine.Grid) : Prop where
  params : grid.params.val = []
  entry : grid.entry = 0#u32
  firstCta : grid.cta0 = 0#u32
  device : grid.device = 0#u32
  pdl : grid.pdl = false
  after : grid.after = none
  explicitCluster : grid.explicit_cluster = false
  preferred : grid.preferred = none
  cgaShape : grid.cga_shape = 0#u32
  waits : grid.waits.val = []

structure LaunchRestrictions (kernel : ptxv_ir.Kernel) (launch : machine.Launch) : Prop where
  kernel : KernelRestrictions kernel
  singleGrid : launch.grids.val.length = 1
  grids : ∀ grid ∈ launch.grids.val, GridRestrictions grid
  noNic : launch.nic = none
  noMulticast : launch.mcast.val = []
  noViews : launch.views.val = []
  oneAddressSpace : launch.spaces.val = []
  noAliased : ∀ selectedAlias ∈ launch.aliased.val, selectedAlias = false
  noHardwareAlias : launch.mc_alias_hw = false
  noTensorMaps : launch.tensormaps.val = []
  noReplicas : launch.replicas.val = []
  noTicketPermutations : launch.perm.val = []
  noNotes : launch.notes.val = []
  noRefutedAssumptions : launch.refuted_opt_in.val = []
  noExitFence : launch.exit_fence = none
  noPollAcquire : launch.poll_acquire = false
  noSharedCumulative : launch.smem_cumulative = false
  noL1 : launch.hw_l1 = false
  noCoreL1 : launch.l1_cores = false
  noL1Havoc : launch.l1_havoc = false
  noReadDone : launch.read_done = false
  noRemoteFifo : launch.remote_fifo = false
  noCollectiveBarrier : launch.hw_bsync_coll = false
  noUniformWindow : launch.hw_split_ur = false
  noHardwareOrder : launch.hw_order = false
  noImplicitTensorFence : launch.tc_fence_implied = false
  noAsyncDestination : launch.async_mb_dest = false
  noResidency : launch.env.residency = none
  noDeviceResidency : launch.env.residency_device = none
  noCooperativeLaunch : launch.env.cooperative = false
  noTensorCotenant : launch.env.tmem_cotenant = false
  noTensorSharing : launch.env.tmem_share.val = []

end MemorySafetyIR
