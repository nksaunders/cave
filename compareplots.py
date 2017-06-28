import numpy as np
import matplotlib.pyplot as pl

x = [1,2,3,4,5]

# 10th mag
m10cdpp = [315.20967841025748, 338.91750472591826, 466.2703120093675, 484.63134426892572, 500.30480294761259]
m10cdppn = [1.2853639701774442, 1.3820398905078886, 1.9013599532949275, 1.9762326838555189, 2.0401460102966826]

# 11th mag
m11cdpp = [445.84291666328073, 451.05992319193638, 537.6077838874528, 541.60906003746652, 540.17527570643813]
m11cdppn = [1.1717027442592469, 1.1854133598998244, 1.4128664876643089, 1.4233820887208375, 1.419614014871162]

# 12th mag
m12cdpp = [637.81563556843287, 617.36846017067558, 680.71054910724308, 685.44862849027743, 660.686867489576]
m12cdppn = [1.0801966873093451, 1.045567603452211, 1.1528429833910474, 1.1608673361480963, 1.1189311233723378]

# 13th mag
m13cdpp = [823.51993040031107, 776.46634763751729, 814.92968708599233, 835.45350532180032, 794.41229576850162]
m13cdppn = [1.0480756418876049, 0.98819158548929242, 1.0371430287919525, 1.0632632393386756, 1.0110309976423417]

# 14th mag
m14cdpp = [987.83546813148746, 914.31356230014694, 943.62011726871913, 955.23347913673308, 937.96020160430248]
m14cdppn = [1.0374214403987652, 0.96020898558310219, 0.9909866296842299, 1.0031829427198167, 0.98504260555213308]

fig1 = pl.figure()
pl.plot(x,m10cdpp,'m', label=r'$\mathrm{K_p\ Mag=10}$')
pl.plot(x,m11cdpp,'k', label=r'$\mathrm{K_p\ Mag=11}$')
pl.plot(x,m12cdpp,'r', label=r'$\mathrm{K_p\ Mag=12}$')
pl.plot(x,m13cdpp,'b', label=r'$\mathrm{K_p\ Mag=13}$')
pl.plot(x,m14cdpp,'g', label=r'$\mathrm{K_p\ Mag=14}$')
pl.title("CDPP vs. Motion Vector Weight")
legend = pl.legend(loc=0)
pl.xlabel("Motion Vector Weight")
pl.ylabel("CDPP")

fig2 = pl.figure()
pl.plot(x,m10cdppn,'m', label=r'$\mathrm{K_p\ Mag=10}$')
pl.plot(x,m11cdppn,'k', label=r'$\mathrm{K_p\ Mag=11}$')
pl.plot(x,m12cdppn,'r', label=r'$\mathrm{K_p\ Mag=12}$')
pl.plot(x,m13cdppn,'b', label=r'$\mathrm{K_p\ Mag=13}$')
pl.plot(x,m14cdppn,'g', label=r'$\mathrm{K_p\ Mag=14}$')
pl.title("Normalized CDPP vs. Motion Vector Weight")
pl.xlabel("Motion Vector Weight")
pl.ylabel("Normalized CDPP")
legend = pl.legend(loc=0)

pl.show()
