import GB300Math.TextureAnisotropyArithmetic

namespace GB300Math.TextureAnisotropyArithmeticWitness

open TextureAnisotropyArithmetic

set_option maxRecDepth 100000
set_option maxHeartbeats 0

def planes : Array (Array (Nat × Nat)) := #[#[(3658246470,12),(805843985,8)],#[(805843985,51),(3034653869,31)],#[(3034653869,51),(2769495153,31)],#[(2769495153,51),(3124349848,31)],#[(3124349848,51),(2905250984,31)],#[(2905250984,51),(700333809,31)],#[(700333809,51),(1324615875,31)],#[(1324615875,51),(1487162298,31)],#[(1487162298,51),(2764467212,31)],#[(2764467212,51),(3292564178,31)],#[(3292564178,51),(3316623318,31)],#[(3316623318,51),(2835329579,31)],#[(2835329579,51),(2737755146,31)],#[(2737755146,12),(3041767498,8)]]

def exactUnits (v : Nat) : Int :=
  let e := exponent v
  let mantissa := v%2^23+(if e=0 then 0 else 2^23)
  let unsigned : Nat := mantissa*2^(e-1)
  if negative v then -(unsigned:Int) else unsigned

def idealUnits (ps : Array (Array (Nat × Nat))) : Int :=
  (ps.flatMap id).foldl (fun acc p => acc+exactUnits p.1*(p.2:Int)) 0

def error (output : Nat) : Nat :=
  (exactUnits output*1024-idealUnits planes).natAbs

def maxInput : Nat := ((planes.flatMap id).map (fun p => (exactUnits p.1).natAbs)).foldl max 0

def sums : Array PlaneSum := #[⟨-1766782080,18,53⟩,⟨-7304248400,-57,-22⟩,⟨-12040071024,-57,-22⟩,⟨-6039434880,-46,-11⟩,⟨-9935844542,-46,-11⟩,⟨-9075565470,-72,-37⟩,⟨7932541392,-5,30⟩,⟨5340588284,15,50⟩,⟨8786108640,15,50⟩,⟨-6258751200,-26,9⟩,⟨-6996917836,-23,12⟩,⟨-9393601056,-23,12⟩,⟨-13676724328,-81,-46⟩,⟨-1725441280,-56,-21⟩]

theorem sums_forward : planes.map planeSum=sums := by decide +kernel

theorem sums_backward : planes.reverse.map planeSum=sums.reverse := by decide +kernel

theorem accumulated_forward : accumulate sums 2=3513154560 := by decide +kernel

theorem accumulated_backward : accumulate sums.reverse 2=3513153984 := by decide +kernel

theorem forward : interpolate planes 4 false=3513154560 := by
  have hn : ((planes).flatMap id).any (fun p => magnitude p.1>0x7f800000)=false := by decide +kernel
  have hi : (((planes).flatMap id).filter (fun p => magnitude p.1=0x7f800000)).isEmpty=true := by decide +kernel
  have hf : sums.all (fun p => p.exponent = -10000)=false := by decide +kernel
  simp only [interpolate, Bool.false_eq_true, ↓reduceIte, hn, hi, hf, Bool.not_true, sums_forward, Id.run_pure]
  rw [show Nat.log2 4=2 by decide +kernel]
  exact accumulated_forward

theorem backward : interpolate planes 4 true=3513153984 := by
  have hn : ((planes.reverse).flatMap id).any (fun p => magnitude p.1>0x7f800000)=false := by decide +kernel
  have hi : (((planes.reverse).flatMap id).filter (fun p => magnitude p.1=0x7f800000)).isEmpty=true := by decide +kernel
  have hf : sums.reverse.all (fun p => p.exponent = -10000)=false := by decide +kernel
  simp only [interpolate, Bool.false_eq_true, ↓reduceIte, hn, hi, hf, Bool.not_true, sums_backward, Id.run_pure]
  rw [show Nat.log2 4=2 by decide +kernel]
  exact accumulated_backward

theorem output_distance : 3513154560-3513153984=576 := by decide +kernel

theorem same_exact_ideal : idealUnits planes=idealUnits planes.reverse := by decide +kernel

theorem forward_error_bound : error 3513154560*10^12 ≤ 301*maxInput*1024 := by decide +kernel

theorem backward_error_bound : error 3513153984*10^12 ≤ 62*maxInput*1024 := by decide +kernel

#print axioms forward
#print axioms backward
#print axioms output_distance
#print axioms same_exact_ideal
#print axioms forward_error_bound
#print axioms backward_error_bound

end GB300Math.TextureAnisotropyArithmeticWitness
