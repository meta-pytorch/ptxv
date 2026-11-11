import SqrtBounds
namespace GB300Math.Log2Drop
open RcpBounds
set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

def bound (u : Nat) : Nat :=
  (1+min u 262142*2)+(4+min u 131068*4)+(16+min u 65528*8)+
  (64+min u 32752*16)+(256+min u 16352*32)+(1024+min u 8128*64)+
  (4096+min u 3968*128)+(16384+min u 1792*256)+(65536+min u 512*512)+262144

theorem square_drop (k u : Nat) (hku : k≤u) : k*k-Fp64Seed.squareTruncated k 19≤bound u := by
  have ha (mask : Nat) : k &&& mask≤min u mask :=
    le_min (le_trans Nat.and_le_left hku) Nat.and_le_right
  have h0:=ha 262142
  have h1:=ha 131068
  have h2:=ha 65528
  have h3:=ha 32752
  have h4:=ha 16352
  have h5:=ha 8128
  have h6:=ha 3968
  have h7:=ha 1792
  have h8:=ha 512
  simp only [Fp64Seed.squareTruncated,Std.Legacy.Range.forIn_eq_forIn_range']
  simp [Std.Legacy.Range.size,List.range',pure_yield_ite,Id.run]
  change k*k≤bound u+(k*k-_)
  simp only [accumulate_three_ite,accumulate_ite]
  have b0 : (if k%2=1 then 1+(k &&& 262142)*2 else 0)≤1+min u 262142*2 := by split <;> omega
  have b1 : (if k>>>1%2=1 then 4+(k &&& 131068)*4 else 0)≤4+min u 131068*4 := by split <;> omega
  have b2 : (if k>>>2%2=1 then 16+(k &&& 65528)*8 else 0)≤16+min u 65528*8 := by split <;> omega
  have b3 : (if k>>>3%2=1 then 64+(k &&& 32752)*16 else 0)≤64+min u 32752*16 := by split <;> omega
  have b4 : (if k>>>4%2=1 then 256+(k &&& 16352)*32 else 0)≤256+min u 16352*32 := by split <;> omega
  have b5 : (if k>>>5%2=1 then 1024+(k &&& 8128)*64 else 0)≤1024+min u 8128*64 := by split <;> omega
  have b6 : (if k>>>6%2=1 then 4096+(k &&& 3968)*128 else 0)≤4096+min u 3968*128 := by split <;> omega
  have b7 : (if k>>>7%2=1 then 16384+(k &&& 1792)*256 else 0)≤16384+min u 1792*256 := by split <;> omega
  have b8 : (if k>>>8%2=1 then 65536+(k &&& 512)*512 else 0)≤65536+min u 512*512 := by split <;> omega
  have b9 : (if k>>>9%2=1 then 262144 else 0)≤262144 := by split <;> omega
  dsimp [bound]
  omega

#print axioms square_drop
end GB300Math.Log2Drop
