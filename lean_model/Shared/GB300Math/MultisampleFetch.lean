import Std

namespace GB300Math.MultisampleFetch

def payload (x y layer sample : Nat) : Nat :=
  1 + x + 16*y + 256*layer + 1024*sample

def encode (n : Nat) : Nat :=
  (n.log2 + 111)*2^23 + (n*2^(23-n.log2)) % 2^23

def word (samples index : Nat) : Nat :=
  if index % 4 = 0 then
    let pixel := index / 4
    encode (payload (pixel % 16) (pixel / 16 % 16)
      (pixel / 256 / samples) (pixel / 256 % samples))
  else if index % 4 = 3 then 0x3f800000 else 0

theorem sample_recovery (x y layer sample : Nat)
    (hx : x < 16) (hy : y < 16) (hl : layer < 2) :
    (payload x y layer sample - 1) / 1024 = sample := by
  unfold payload
  omega

theorem layer_recovery (x y layer sample : Nat)
    (hx : x < 16) (hy : y < 16) (hl : layer < 2) :
    (payload x y layer sample - 1) / 256 % 4 = layer := by
  unfold payload
  omega

theorem pixel_recovery (x y layer sample : Nat)
    (hx : x < 16) (hy : y < 16) :
    (payload x y layer sample - 1) % 16 = x ∧
    (payload x y layer sample - 1) / 16 % 16 = y := by
  unfold payload
  omega

theorem first_payload : word 2 0 = 0x37800000 := by decide +kernel

theorem last_payload : encode (payload 15 15 1 7) = 0x3df00000 := by decide +kernel

theorem sample_is_distinct : word 2 0 ≠ word 2 1024 := by decide +kernel

theorem layer_is_distinct : word 2 0 ≠ word 2 2048 := by decide +kernel

theorem alpha_one (samples pixel : Nat) :
    word samples (4*pixel+3) = 0x3f800000 := by
  simp [word, Nat.add_mod]

end GB300Math.MultisampleFetch
