import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import h5py

lenses = h5py.File('../cats/DESY3/desy3_redmapper_v6.4.22+2_release.h5')
path_cl = 'catalog/cluster/'
path_mem = 'catalog/cluster_members/'

redshift = 'z_lambda'
idcol = 'mem_match_id'
lambdaz = 'lambda_chisq'

idx = 0

mask = (lenses[path_cl+redshift][:]>0.19) & (lenses[path_cl+redshift][:]<0.27) & (lenses[path_cl+lambdaz][:]>38.0) & (lenses[path_cl+lambdaz][:]<55.0)

cl_id = np.sort(lenses[path_cl][idcol][mask].data)

mask_cl = lenses[path_cl+idcol][:] == cl_id[idx]
mask_mem = lenses[path_mem+idcol][:] == cl_id[idx]

n_mem = np.sum(mask_mem)

# center of the cluster : corresponds to the BGC position
ra_cl = lenses[path_cl]['ra'][mask_cl]
dec_cl = lenses[path_cl]['dec'][mask_cl]

# member positions
ra_mem = lenses[path_mem]['ra'][mask_mem]
dec_mem = lenses[path_mem]['dec'][mask_mem]
p_mem = lenses[path_mem]['pmem'][mask_mem]

# member positions wrt BCG position
# excluding the BGC for analysis
bgc_idx = np.where(ra_mem==ra_cl)[0][0]
assert np.where(dec_mem==dec_cl)[0][0] == bgc_idx
x_m = np.delete(ra_mem-ra_cl, bgc_idx) 
y_m = np.delete(dec_mem-dec_cl, bgc_idx)
p_mem = np.delete(p_mem, bgc_idx) 
r_m = np.hypot(x_m,y_m)

## should the components be weighted by the pmem? or magnitude or something?
inertia = np.zeros((2,2))
inertia[0,0] = np.sum(x_m**2)/n_mem
inertia[1,1] = np.sum(y_m**2)/n_mem
inertia[0,1] = np.sum(x_m*y_m)/n_mem
inertia[1,0] = inertia[0,1]

inertia_r = np.zeros((2,2))
inertia_r[0,0] = np.sum(x_m**2/r_m**2)/n_mem
inertia_r[1,1] = np.sum(y_m**2/r_m**2)/n_mem
inertia_r[0,1] = np.sum(x_m*y_m/r_m**2)/n_mem
inertia_r[1,0] = inertia_r[0,1]

inertia_m = np.zeros((2,2))
inertia_m[0,0] = np.sum(p_mem*x_m**2)
inertia_m[1,1] = np.sum(p_mem*y_m**2)
inertia_m[0,1] = np.sum(p_mem*y_m*x_m)
inertia_m[1,0] = inertia_m[0,1]

eigval, eigvec = np.linalg.eigh(inertia)
eigval_r, eigvec_r = np.linalg.eigh(inertia_r)
eigval_m, eigvec_m = np.linalg.eigh(inertia_m)

q = eigval[0]/eigval[1]
q_r = eigval_r[0]/eigval_r[1]
q_m = eigval_m[0]/eigval_m[1]

print(f'{q=:.2f}')
print(f'{q_r=:.2f}')
print(f'{q_m=:.2f}')

plt.figure()
plt.scatter(x_m, y_m, s=15, edgecolors='C0', facecolor='none')
for i, lsi in zip(range(2), ['-', '--']):
    plt.axline([0.0,0.0], *[eigvec[i]*eigval[i]], c='b', ls=lsi)
    plt.axline([0.0,0.0], *[eigvec_r[i]*eigval_r[i]], c='r', ls=lsi)
    plt.axline([0.0,0.0], *[eigvec_m[i]*eigval_m[i]], c='limegreen', ls=lsi)
plt.axvline(0.0, c='gray', ls=':', alpha=0.3)
plt.axhline(0.0, c='gray', ls=':', alpha=0.3)
plt.plot([],[],c='b',label='std', alpha=0.5)
plt.plot([],[],c='r',label='red', alpha=0.5)
plt.plot([],[],c='limegreen',label='mass', alpha=0.5)
plt.legend()
plt.show()
