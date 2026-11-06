import GB300Math.Mufu
import Lean.Elab.Tactic.Omega
namespace GB300Math.TanhNativeCore
set_option maxRecDepth 100000
set_option maxHeartbeats 100000

def value (e f:Nat) : Int :=
  let table := TanhTables.binades[e]!
  let k := f/2^table.1%2^17
  let cs := table.2[f/2^table.1/2^17]!
  cs.1*2^34+cs.2.1*(k:Int)*2^17+cs.2.2*(Fp64Seed.squareTruncated k 19:Int)

theorem regroup (g:Int→Nat) (table:Nat×Array (Int×Int×Int)) (f s:Nat) :
    Mufu.tanh.match_1 (fun _=>Option Nat) table (fun d cs=>
      Mufu.interpolate.match_1 (fun _=>Option Nat) cs[f/2^d/2^17]! (fun a b c=>
        some (s+g (a*2^34+b*((f/2^d%2^17:Nat):Int)*2^17+c*(Fp64Seed.squareTruncated (f/2^d%2^17) 19:Int))))) =
    some (s+g (
      (table.2[f/2^table.1/2^17]!).1*2^34+
      (table.2[f/2^table.1/2^17]!).2.1*((f/2^table.1%2^17:Nat):Int)*2^17+
      (table.2[f/2^table.1/2^17]!).2.2*(Fp64Seed.squareTruncated (f/2^table.1%2^17) 19:Int))) := by
  rcases table with ⟨d,cs⟩
  dsimp only [Mufu.tanh.match_1]
  rcases cs[f/2^d/2^17]! with ⟨a,b,c⟩
  rfl

theorem central (x:Nat) (hmag:Mufu.magnitude x≤0x7f800000)
    (hlo:119≤Mufu.exponent x) (hhi:Mufu.exponent x<130) :
    Mufu.tanh x= some (Mufu.sign x+Mufu.fixedToWord
      (value (Mufu.exponent x-119) (Mufu.fraction x)).toNat
      (184-(min (Mufu.exponent x) 126:Int))) := by
  unfold Mufu.tanh
  rw [ite_eq_right (by omega),ite_eq_right (by omega),ite_eq_right (by omega)]
  exact regroup (fun v=>Mufu.fixedToWord v.toNat (184-(min (Mufu.exponent x) 126:Int)))
    (TanhTables.binades[Mufu.exponent x-119]!) (Mufu.fraction x) (Mufu.sign x)
#print axioms central
end GB300Math.TanhNativeCore
