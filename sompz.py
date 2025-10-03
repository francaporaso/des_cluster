import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.table import Table

sompz = h5py.File('../cats/DESY3/desy3_sompz_v050.h5')
sompz = sompz['catalog/sompz']

metacal = Table.read('../cats/DESY3/desy3_unsheared.fits', memmap=True, format='fits')
metacal = metacal[metacal['unsheared_flags']==0]

flag_mask = np.isin(sompz['unsheared/coadd_object_id'], metacal['coadd_object_id'])
bin0_mask = sompz['unsheared/bhat'][:] == 0

cellid_bin0 = sompz['unsheared/cell_wide'][flag_mask&bin0_mask]

pz_bin0 = np.zeros(601)
for i, cid in enumerate(cellid_bin0):
    pz_bin0 += sompz['pzdata/pz_chat'][:][cid]

zrange = (sompz['pzdata/zlow'][:]+sompz['pzdata/zhigh'][:])*0.5
plt.plot(zrange, pz_bin0)
plt.savefig('bin0_test.png')

if __name__ == '__main__':
    print('a')
