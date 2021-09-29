import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('learning-curve.out', skip_header=22)

epoch = data[:,0]
e_train = data[:,1]*1000
e_valid = data[:,2]*1000

f_train = data[:,9]
f_valid = data[:,10]

f_frac = 1

loss_t = e_train/1000+f_frac*f_train
loss_v = e_valid/1000 + f_frac*f_valid

low_ndx = np.where(loss_v==min(loss_v))
print(low_ndx)
print('E(T) rmse = {}'.format(e_train[low_ndx][0]))
print('E(V) rmse = {}'.format(e_valid[low_ndx][0]))
print('F(T) rmse = {}'.format(f_train[low_ndx][0]))
print('F(V) rmse = {}'.format(f_valid[low_ndx][0]))
    
plt.plot(epoch, e_train)
plt.plot(epoch, e_valid)
plt.ylim([0,5])
# plt.xlim([0, 20])
plt.legend(['Train','Valid'])
plt.ylabel('E/atom (meV/atom)')
plt.xlabel('Epoch')

plt.figure()
plt.plot(epoch, f_train)
plt.plot(epoch, f_valid)
plt.ylim([0.2,1])
# plt.xlim([0, 20])
plt.legend(['Train','Valid'])
plt.ylabel('F (eV/atom)')
plt.xlabel('Epoch')

plt.figure()
plt.plot(epoch, loss_t)
plt.plot(epoch, loss_v)
# plt.ylim([0.2,1])
# plt.xlim([0, 20])
plt.legend(['Train','Valid'])
plt.ylabel('Loss Function')
plt.xlabel('Epoch')
plt.show()
