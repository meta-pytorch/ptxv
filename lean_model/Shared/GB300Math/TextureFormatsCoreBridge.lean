import GB300Math.TextureFormatsCore
import TextureFormats

namespace GB300Math.TextureFormatsCore

theorem evaluate_matches (format mode value : Nat) :
    evaluate format mode value = TextureFormats.evaluate format mode value := by rfl

theorem width_matches (format : Nat) : width format = TextureFormats.width format := by rfl

end GB300Math.TextureFormatsCore
#print axioms GB300Math.TextureFormatsCore.evaluate_matches
#print axioms GB300Math.TextureFormatsCore.width_matches
