import numpy as np
# import matplotlib.pyplot as plt

train_ndx = np.loadtxt('train.txt')
valid_ndx = np.loadtxt('valid.txt')
train_ndx = [int(train_ndx[i]) for i in range(len(train_ndx))]
valid_ndx = [int(valid_ndx[i]) for i in range(len(valid_ndx))]

print(len(train_ndx) + len(valid_ndx))

data_e = np.genfromtxt('energy.comp', skip_header=13)
natoms = data_e[:,1]
edft = data_e[:,2]
enn = data_e[:,3]
edft = edft/natoms
enn = enn/natoms
edft_t = np.take(edft, train_ndx)
edft_v = np.take(edft, valid_ndx)
enn_t = np.take(enn, train_ndx)
enn_v = np.take(enn, valid_ndx)
ee = abs(edft-enn)*1000
ee_t = np.take(ee, train_ndx)
ee_v = np.take(ee, valid_ndx)
e_rmse_t = np.sqrt(np.mean(ee_t**2))
e_rmse_v = np.sqrt(np.mean(ee_v**2))

e_all = np.concatenate((edft,enn))
e_min = np.min(np.min(e_all))
e_max = np.max(np.max(e_all))
xx = [e_min,e_max]
x_text = np.mean(xx)-0.05*(e_max-e_min)
y_text = np.mean(xx)-0.3*(e_max-e_min)

print('==========energy calculations done===========')

np.savetxt('edft_v.txt',edft_v)
np.savetxt('edft_t.txt',edft_t)
np.savetxt('enn_v.txt',enn_v)
np.savetxt('enn_t.txt',enn_t)

# plt.figure(figsize=(10,7.5))
# plt.subplot(1, 2, 1)
# plt.plot(edft_v,enn_v,'o', color='#1f77b4')
# plt.plot(edft_t,enn_t,'o', color='#ff7f0e')
# plt.plot(xx,xx,'k--')
# plt.text(x_text,y_text,'RMSE(T) = %.02f meV/atom'% (e_rmse_t))
# plt.text(x_text,y_text-0.1*(e_max-e_min),'RMSE(V) = %.02f meV/atom'% (e_rmse_v))
# plt.xlabel('$E_{DFT}$ eV/atom')
# plt.ylabel('$E_{NN}$ eV/atom')

data_f = np.genfromtxt('forces.comp', skip_header=13)
fdft = data_f[:,2]
fnn = data_f[:,3]
struc_ndx = data_f[:,0]

ndx_t = []
ndx_v = []

for j,i in enumerate(struc_ndx):
    if i in train_ndx:
        ndx_t.append(j)
    else:
        ndx_v.append(j)

fdft_t = np.take(fdft, ndx_t)
fdft_v = np.take(fdft, ndx_v)
fnn_t = np.take(fnn, ndx_t)
fnn_v = np.take(fnn, ndx_v)
fe = abs(fdft-fnn)
fe_t = np.take(fe, ndx_t)
fe_v = np.take(fe, ndx_v)
f_rmse_t = np.sqrt(np.mean(fe_t**2))
f_rmse_v = np.sqrt(np.mean(fe_v**2))

np.savetxt('fdft_t.txt',fdft_t)
np.savetxt('fdft_v.txt',fdft_v)
np.savetxt('fnn_t.txt',fnn_t)
np.savetxt('fnn_v.txt',fnn_v)

# print('==========force calculations done===========')

# f_all = np.concatenate((fdft,fnn))
# f_min = np.min(np.min(f_all))
# f_max = np.max(np.max(f_all))
# xx = [f_min,f_max]
# x_text = np.mean(xx)-0.05*(f_max-f_min)
# y_text = np.mean(xx)-0.3*(f_max-f_min)

# plt.subplot(1, 2, 2)
# plt.plot(fdft_v,fnn_v,'o', color='#1f77b4')
# plt.plot(fdft_t,fnn_t,'o', color='#ff7f0e')

# plt.plot(xx,xx,'--')
# plt.text(x_text,y_text,'RMSE(T) = %.02f eV/$\AA$'% (f_rmse_t))
# plt.text(x_text,y_text-0.1*(f_max-f_min),'RMSE(V) = %.02f eV/$\AA$'% (f_rmse_v))
# plt.xlim([-3,2])
# plt.xlabel('$F_{DFT}$ eV/$\AA$')
# plt.ylabel('$F_{NN}$ eV/$\AA$')

# plt.show()
# plt.savefig('parity.png', dpi=1000)
