import GB300Math.SmallConversions

open GB300Math.SmallConversions

def read32 (data : ByteArray) (offset : Nat) : Nat := readWord data offset 4

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected scaled-BF16 manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [relu, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed scaled-BF16 row")
    unless relu = "0" ∨ relu = "1" do throw (IO.userError "Invalid RELU flag")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/16*8208 do
      throw (IO.userError "Invalid capture sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let base := read32 xs (16*i)
      let words := read32 xs (16*i+4)
      let pairing := read32 xs (16*i+8)
      let scale := base/65536
      unless scale < 256 ∧ base % 65536 = 0 ∧ words = 65536 ∧ pairing < 2 ∧ read32 xs (16*i+12) = 0 do
        throw (IO.userError "Invalid scale partition")
      let start := 8208*i
      let count := read32 ys (start+8)
      unless count ≤ 1024 ∧ read32 ys (start+12) = 0 do
        throw (IO.userError "Transition overflow")
      let mut previous := 0
      for j in [:1024] do
        let offset := read32 ys (start+16+4*j)
        let value := read32 ys (start+4112+4*j)
        if j < count then
          unless previous < offset ∧ offset < 65536 ∧ value < 65536 do
            throw (IO.userError "Invalid transition")
          previous := offset
        else unless offset = 0 ∧ value = 0 do
          throw (IO.userError "Nonzero unused transition slot")
      let mut next := 0
      let mut actual := read32 ys start
      for word in [:65536] do
        if next < count ∧ word = read32 ys (start+16+4*next) then
          actual := read32 ys (start+4112+4*next)
          next := next+1
        let hi := downS2F6 true (relu = "1") true (word*65536) scale
        let lo := if pairing = 0 then hi else downS2F6 true (relu = "1") true ((65535-word)*65536) scale
        let target := hi*256+lo
        if actual ≠ target then
          if errors < 3 then IO.println s!"witness scale={scale} word={word} pairing={pairing} expected={target} actual={actual}"
          errors := errors+1
      unless actual = read32 ys (start+4) ∧ next = count do
        throw (IO.userError "Incorrect terminal transition")
    IO.println s!"result {output} records={xs.size/16*65536} mismatches={errors} expected={expected}"
    if errors ≠ expected then failures := failures+1
  return if failures = 0 then 0 else 1
