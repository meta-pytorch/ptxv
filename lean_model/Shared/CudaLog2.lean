/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

set_option maxRecDepth 100000
set_option maxHeartbeats 20000000

namespace GB300Math.CudaLog2

open FloatLib.Floats

abbrev Word := BitVec 32
abbrev F32 := ExecFloat.Binary 8 23

def value (x : Word) : F32 := ExecFloat.Binary.ofNatBits x.toNat

def word (x : F32) : Word :=
  let bits := BitVec.ofNat 32 (ExecFloat.Binary.toNatBits x)
  if bits &&& 0x7fffffff > 0x7f800000 then 0x7fffffff else bits

def add (x y : Word) : Word := word (value x + value y)
def mul (x y : Word) : Word := word (value x * value y)
def fma (x y z : Word) : Word := word (ExecFloat.fma (value x) (value y) (value z))
def lt (x y : Word) : Bool := decide (value x < value y)
def eq (x y : Word) : Bool := value x == value y

def fromSigned (x : Word) : Word :=
  let n := x.toInt
  let positive : F32 := OfScientific.ofScientific n.natAbs false 0
  word (if n < 0 then -positive else positive)

/-- Manual transcription of the retained CUDA 13.1 PTX scalar body, registers r6–r31. -/
def model (r6 : Word) : Word :=
  let p2 := lt r6 0x00800000
  let r7 := mul r6 0x4b000000
  let r8 := if p2 then r7 else r6
  let r9 : Word := if p2 then 0xc1b80000 else 0
  let r10 := r8 - 1060439283
  let r11 := r10 &&& 0xff800000
  let r12 := r8 - r11
  let r13 := fromSigned r11
  let r14 := fma r13 0x34000000 r9
  let r15 := add r12 0xbf800000
  let r16 := fma r15 0x3dc6b27f 0xbe2c7f30
  let r17 := fma r16 r15 0x3e2fcf2a
  let r18 := fma r17 r15 0xbe374e43
  let r19 := fma r18 r15 0x3e520bf4
  let r20 := fma r19 r15 0xbe763c8b
  let r21 := fma r20 r15 0x3e93bf99
  let r22 := fma r21 r15 0xbeb8aa49
  let r23 := fma r22 r15 0x3ef6384a
  let r24 := fma r23 r15 0xbf38aa3b
  let r25 := mul r15 r24
  let r26 := mul r15 r25
  let r27 := fma r15 0x3fb8aa3b r26
  let r28 := add r14 r27
  let p3 := r8 > 2139095039
  let r29 := fma r8 0x7f800000 0x7f800000
  let r30 := if p3 then r29 else r28
  if eq r8 0 then 0xff800000 else r30

def readWord (bytes : ByteArray) (index : Nat) : Word :=
  let offset := index * 4
  BitVec.ofNat 32 (bytes[offset]!.toNat + bytes[offset + 1]!.toNat * 256 +
    bytes[offset + 2]!.toNat * 65536 + bytes[offset + 3]!.toNat * 16777216)

theorem at_one : model 0x3f800000 = 0 := by decide +kernel

theorem at_next_one : model 0x3f800001 = 0x3438aa3a := by decide +kernel

#print axioms at_one
#print axioms at_next_one

end GB300Math.CudaLog2

open GB300Math.CudaLog2

def main (args : List String) : IO UInt32 := do
  let [inputPath, outputPath, strideString] := args |
    throw (IO.userError "Usage: CudaLog2 <inputs.u32> <cuda_log2.u32> <stride>")
  let some stride := strideString.toNat? | throw (IO.userError "Invalid stride")
  unless stride > 0 do throw (IO.userError "Stride must be positive")
  let inputs ← IO.FS.readBinFile inputPath
  let outputs ← IO.FS.readBinFile outputPath
  unless inputs.size > 0 ∧ inputs.size = outputs.size ∧ inputs.size % 4 = 0 do
    throw (IO.userError "Invalid input/output byte counts")
  let mut mismatches := 0
  let mut checked := 0
  for index in [:inputs.size / 4] do
    if index % stride != 0 then continue
    let input := readWord inputs index
    let expected := model input
    let actual := readWord outputs index
    if expected != actual then
      if mismatches < 8 then
        IO.eprintln s!"input={input.toNat} model={expected.toNat} device={actual.toNat}"
      mismatches := mismatches + 1
    checked := checked + 1
  IO.println s!"cuda_log2 checked={checked} stride={stride} mismatches={mismatches}"
  return if mismatches = 0 then 0 else 1
