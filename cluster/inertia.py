import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.wcs import WCS
import h5py

COSMO = FlatLambdaCDM(H0=100.0, Om0=0.3)

ZMIN, ZMAX = 0.19, 0.27
LMIN, LMAX = 38.0, 55.0

CLUSTERS = Table.read('../../cats/DESY3/desy3_redmapper_cluster-ws.fits', format='fits', memmap=True)
MEMBERS = Table.read('../../cats/DESY3/desy3_redmapper_cluster-members.fits', format='fits', memmap=True)

def gnomonic_projection(ra, dec, ra0:float, dec0:float):
    '''
    gnomonic projection of the shpere on a tangent point at (ra0,dec0).
    Maps spherical coordinates to a tangent plane.
    '''
    ra = np.deg2rad(ra-ra0)
    dec = np.deg2rad(dec-dec0)

    x = np.sin(ra)/np.cos(ra)
    y = np.sin(dec)/(np.cos(dec)*np.cos(ra))

    return x, y

def inertia(dx,dy,weight):
    '''
    "momentos" func from github.com/elizabethjg/multipole_density_profile/blob/master/member_distribution.py
    Equations 2, 3 & 4 of Gonzalez et al 2020 (arxiv.org/pdf/2006.08651)
    '''

    Q11  = np.sum((dx**2)*weight)/np.sum(weight)
    Q22  = np.sum((dy**2)*weight)/np.sum(weight)
    Q12  = np.sum((dx*dy)*weight)/np.sum(weight)
    E1 = (Q11-Q22)/(Q11+Q22)
    E2 = (2.*Q12)/(Q11+Q22)
    e = np.sqrt(E1**2 + E2**2)
    theta = np.arctan2(E2,E1)/2.0
    return e, theta

def cluster_orientation(ra_cl, dec_cl, ra_mem, dec_mem, weight=None):
    if not weight:
        weight = np.ones_like(ra_mem)
    x, y = gnomonic_projection(ra_mem, dec_mem, ra_cl, dec_cl)
    e, theta = inertia(x, y, weight)
    return x, y, e, theta

def main():

    wcs = WCS(naxis=2)
    wcs.wcs.crpix=[0.0, 0.0]
    wcs.wcs.cdelt=[1.0/3600.0, 1.0/3600.0]
    wcs.wcs.cunit=['deg', 'deg']
    wcs.wcs.ctype=['RA---TAN', 'DEC--TAN']

    l = CLUSTERS[ (CLUSTERS['lambda']>LMIN) & (CLUSTERS['lambda']<=LMAX) & (CLUSTERS['redshift']>ZMIN) & (CLUSTERS['redshift']<=ZMAX) ]
    # l = CLUSTERS
    n_cl = len(l)
    id_cl = l['mem_match_id'].data

    samples_names = ['allmem', 'pmemcut']
    weights_names = ['wo_weight', 'lum', 'dist']

    results = {sam: {
        w: {'idx': np.zeros(n_cl, dtype=int), 
            'e': np.zeros(n_cl, dtype=float), 
            'theta': np.zeros(n_cl, dtype=float)}
            for w in weights_names
        } for sam in samples_names
    }

    for i, idx in enumerate(id_cl):
        l_idx = l[id_cl==idx]

        mask_mem = MEMBERS['mem_match_id']==idx
        mask_cen = MEMBERS['ra_mem']!=l_idx['ra_cl']
        m = MEMBERS[mask_mem&mask_cen]

        for sam in samples_names:

            if sam=='pmemcut':
                m = m[m['pmem']>0.5]

            wcs.wcs.crval = list(l_idx['ra_cl', 'dec_cl'].as_array()[0])
            x, y = wcs.wcs_world2pix(m['ra_mem'], m['dec_mem'], 0)

            #plt.scatter(x,y,s=10,edgecolor=f'C{i}',facecolor=f'C{i}' if sam=='pmemcut' else 'none',label=i)

            mag_abs_r = m['model_mag_r'].data+5.0-5.0*np.log10(COSMO.luminosity_distance(l_idx['redshift']).to('pc').value)
            lum_r = 10.0**(-0.4*mag_abs_r)

            weights = {
                w: v for w,v in zip(weights_names, [np.ones(len(m)), lum_r, 1.0/np.hypot(x,y)])
            }

            for w in weights_names:
                results[sam][w]['idx'][i] = idx
                e, theta = inertia(x, y, weight=weights[w])
                results[sam][w]['e'][i] = e
                results[sam][w]['theta'][i] = theta
                #plt.axline([0.,0.],slope=np.tan(theta[sam][w][i]),ls='-' if sam=='pmemcut' else '--', c=f'C{i}')
    #plt.legend(bbox_to_anchor=(1.5,1.5), ncol=5)

    with h5py.File('test-orientations.h5', 'w') as f:
        for sam, weights in results.items():
            grp_sam = f.create_group(sam)
            
            for w, data in weights.items():
                grp_w = grp_sam.create_group(w)
                
                for key, array in data.items():
                    grp_w.create_dataset(key, data=array)

    # np.savetxt('../cats/DESY3/redmapper_orientation.dat', np.vstack([id_cl, e, theta]), header='# id_cl | e | theta ')

if __name__=='__main__':
    print('Start...')
    main()
    print('End!')