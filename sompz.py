import numpy as np
import matplotlib.pyplot as plt
import h5py

def open_sompz():
    f = h5py.File('desy3_sompz_v050.h5')
    df = f['catalog']['sompz']

    # sompz.keys() = ['pzdata','sheared_1m','sheared_1p','sheared_2m','sheared_2p']
    ## pzdata.keys() = ['bin0','bin1','bin2','bin3','pz_c','pz_chat','z_high','z_low']
    return df

if __name__ = '__main__':
    print('a')
