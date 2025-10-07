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

    return pz_chat, pchat_s

if __name__ == '__main__':
    tin = time.time()
    pz_chat, pchat_s = main()

    pz_bin = np.array([
        np.sum([pz_chat[j][i]*pchat_s[j][i] for i in range(len(pchat_s[j]))])
        for j in range(4)
    ])
    np.savetxt('test_mcal_sompz.dat', pz_bin)
    # t = Table({
    #     'pz_chat_bin0':pz_chat[0],
    #     'pz_chat_bin1':pz_chat[1],
    #     'pz_chat_bin2':pz_chat[2],
    #     'pz_chat_bin3':pz_chat[3],
    #     'pchat_s_bin0':pchat_s[0],
    #     'pchat_s_bin1':pchat_s[1],
    #     'pchat_s_bin2':pchat_s[2],
    #     'pchat_s_bin3':pchat_s[3],

    # })
    # t.write('test_mcal_sompz.dat')
    print(f'took {time.time()-tin} s')
