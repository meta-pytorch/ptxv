import MultimemReduction
set_option maxHeartbeats 0
set_option maxRecDepth 100000
namespace GB300Math.MultimemReduction

theorem e4m3_extended_finite_code_is_ignored :
    candidates "max" "e4m3" "default" [0, 0x7e] = some [0] := by decide +kernel

theorem e4m3_code_78_is_retained :
    candidates "max" "e4m3" "default" [0, 0x78] = some [0x78] := by decide +kernel

theorem e4m3_quiet_payload_set :
    candidates "min" "e4m3" "default" [0x79, 0xfa] = some [0x7d, 0xfe] := by decide +kernel

theorem e5m2_infinity_code_is_retained :
    candidates "max" "e5m2" "default" [0, 0x7c] = some [0x7c] := by decide +kernel

theorem e5m2_max_prefers_positive_zero :
    candidates "max" "e5m2" "default" [0x80, 0] = some [0] := by decide +kernel

theorem e5m2_min_prefers_negative_zero :
    candidates "min" "e5m2" "default" [0x80, 0] = some [0x80] := by decide +kernel

#print axioms e4m3_extended_finite_code_is_ignored
#print axioms e4m3_code_78_is_retained
#print axioms e4m3_quiet_payload_set
#print axioms e5m2_infinity_code_is_retained
#print axioms e5m2_max_prefers_positive_zero
#print axioms e5m2_min_prefers_negative_zero
end GB300Math.MultimemReduction
