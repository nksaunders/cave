import numpy as np
import matplotlib.pyplot as pl

x = [1,2,3,4,5]

# 11th mag
m11cdpp = [445.84291666328073, 451.05992319193638, 537.6077838874528, 541.60906003746652, 540.17527570643813]
m11cdppn = [1.1717027442592469, 1.1854133598998244, 1.4128664876643089, 1.4233820887208375, 1.419614014871162]

# 12th mag
m12cdpp = [637.81563556843287, 617.36846017067558, 680.71054910724308, 685.44862849027743, 660.686867489576]
m12cdppn = [1.0801966873093451, 1.045567603452211, 1.1528429833910474, 1.1608673361480963, 1.1189311233723378]

# 13th mag


fig1 = pl.figure()
pl.plot(x,m11cdpp,'k', label='Kp=11')
pl.plot(x,m12cdpp,'r', label='Kp=12')
pl.plot(x,m13cdpp,'b', label='Kp=13')
# pl.plot(x,m14cdpp,'g')
pl.title("CDPP vs. Motion Magnitude")
legend = pl.legend()
pl.xlabel("Motion Magnitude")
pl.ylabel("CDPP")
pl.show()

fig2 = pl.figure()
pl.plot(x,m11cdppn,'k', label='Kp=11')
pl.plot(x,m12cdppn,'r', label='Kp=12')
pl.plot(x,m13cdppn,'b', label='Kp=13')
# pl.plot(x,m14cdppn,'g')
pl.title("Normalized CDPP vs. Motion Magnitude")
pl.xlabel("Motion Magnitude")
pl.ylabel("Normalized CDPP")
legend = pl.legend()

pl.show()
