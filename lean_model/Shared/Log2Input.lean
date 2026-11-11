import Root32FloatLib
import Log2Table
namespace GB300Math.Log2Input
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxHeartbeats 3000000

def input (e : Fin 254) (f : Fin (2^23)) : Rat :=
  (1+(f.val:Rat)/2^23)*(2:Rat)^((e.val:Int)-126)

theorem decoded (e : Fin 254) (f : Fin (2^23)) :
    Root32FloatLib.value ((1+e.val)*2^23+f.val)=some (input e f) := by
  have h := MufuRcp32Bounds.decode_normal e f
  have he : (e.val:Int)-149=((e.val:Int)-126)-23 := by omega
  rw [he,zpow_sub₀ (by norm_num : (2:Rat)≠0)] at h
  unfold Root32FloatLib.value
  convert h using 1
  congr 1
  dsimp [input]
  push_cast
  norm_num
  ring

theorem positive (e : Fin 254) (f : Fin (2^23)) : 0 < input e f := by
  dsimp [input]
  positivity

theorem ideal (e : Fin 254) (f : Fin (2^23)) :
    Real.log (input e f:ℝ)/Real.log 2=
      (e.val:ℝ)-126+Real.log (1+(f.val:ℝ)/2^23)/Real.log 2 := by
  have hlog := Log2NearOne.log_two_interval
  have hlogne : Real.log 2≠0 := by linarith
  have hm : 0<(1+(f.val:ℝ)/2^23) := by positivity
  dsimp [input]
  push_cast
  rw [Real.log_mul (ne_of_gt hm) (by positivity),Real.log_zpow]
  push_cast
  field_simp
  ring

#print axioms decoded
#print axioms positive
#print axioms ideal
end GB300Math.Log2Input
