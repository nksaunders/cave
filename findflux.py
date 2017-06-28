from everest.math import SavGol
import everest
import numpy as np

ID = 212278696

# 9th: 212273934
# 10th: 212278696

star = everest.Everest(ID)
time = star.apply_mask(star.time)
flux = star.apply_mask(star.flux)

sgflux = SavGol(flux)

print(np.mean(sgflux))
