import numpy as np
import h5py
from astropy.table import Table
import time
import pandas as pd

def main():
    sompz = h5py.File('../cats/DESY3/desy3_sompz_v050.h5')
    sompz = sompz['catalog/sompz']

    metacal = Table.read('../cats/DESY3/desy3_unsheared.fits', memmap=True, format='fits')
    metacal = metacal[metacal['unsheared_flags']==0]
    j = np.sort(metacal['coadd_object_id'])

    mask = np.isin(sompz['unsheared/coadd_object_id'], j)
    df = pd.DataFrame({
        'bhat':sompz['unsheared/bhat'][mask],
        'cell_wide':sompz['unsheared/cell_wide'][mask],
    })
    chat_bin=[]
    for i in range(4):
        chat_bin.append(np.unique(df.query(f'bhat=={i}')['cell_wide'], return_counts=True))

    pchat_s = [chat_bin[i][1]/len(j) for i in range(4)]
    pz_chat = [sompz['pzdata/pz_chat'][chat_bin[i][0]] for i in range(4)]

    np.savetxt('test_mcal_sompz.dat', np.hstack(pchat_s, pz_chat))
    

if __name__ == '__main__':
    tin = time.time()
    main()
    print(f'took {time.time()-tin} s')
