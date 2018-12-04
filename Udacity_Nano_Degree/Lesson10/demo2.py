import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


r= np.linspace(-2.1,2.1,300)
s,t = np.meshgrid(r,r)
s = np.reshape(s,(np.size(s),1))
t = np.reshape(t,(np.size(t),1))
h = np.concatenate((s,t),1)

print("s size = ",s.shape)
print("t size = ",t.shape)
print("h size = ",h.shape)
print(s[:5,:])
print(t[:5,:])
print(h[:5,:])

# z = clf.predict(h)
s = s.reshape((np.size(r),np.size(r)))
t = t.reshape((np.size(r),np.size(r)))
print("s size = ",s.shape)
print("t size = ",t.shape)
# z = z.reshape((np.size(r),np.size(r)))
# plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))