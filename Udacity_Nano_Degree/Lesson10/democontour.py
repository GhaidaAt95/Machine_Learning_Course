import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm




# ax = fig.add_subplot(111)
# u = np.linspace(-1,1,100)
# x, y = np.meshgrid(u,u)
# z = x**2 + y**2

# print("U\n",u.shape)
# print("X\n",x.shape)
# print("y\n",y.shape)

# ax.contour(x,y,z)
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

xvalues = np.array(np.linspace(-10,10,10));
yvalues = np.array(np.linspace(-10,10,10));

xx, yy = np.meshgrid(xvalues, yvalues)
print(xx.shape," && ",yy.shape)
print((xx**2+yy**2).shape)
ax.plot(xx, yy, marker='.', color='k', linestyle='none')

z2 = xx + yy
ax2.contourf(xx,yy,z2)

z3 = xx**2+yy**2
ax3.set_xlim(-15,15)
ax3.set_ylim(-15,15)
ax3.contourf(xx,yy,z3,colors=['blue','red'],levels=[0,25,125])

ax4.set_xlim(-15,15)
ax4.set_ylim(-15,15)
ax4 = plt.contour(xx,yy,z3,colors=['blue','red'])
plt.clabel(ax4,fmt='%2.1f', colors='k', fontsize=7)
plt.show()

