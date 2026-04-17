import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import h5py

ZMIN, ZMAX = 0.19, 0.27
LMIN, LMAX = 38.0, 55.0

CLUSTERS = Table.read('../cats/DESY3/desy3_redmapper_cluster-ws.fits', format='fits', memmap=True)
MEMBERS = Table.read('../cats/DESY3/desy3_redmapper_cluster-members.fits', format='fits', memmap=True)

def gnomonic_projection(ra, dec, ra0, dec0):
    '''
    gnomonic projection of the shpere on a tangent point at (ra0,dec0).
    Maps spherical coordinates to a tangent plane.
    '''
    ra = np.deg2rad(ra-ra0)
    dec = np.deg2rad(dec-dec0)

    x = np.sin(ra)/np.cos(ra)
    y = np.sin(dec)/(np.cos(dec)*np.cos(ra))

    return x, y

def inertia(dx,dy,w):
    '''
    "momentos" func from github.com/elizabethjg/multipole_density_profile/blob/master/member_distribution.py
    Equations 2, 3 & 4 of Gonzalez et al 2020 (arxiv.org/pdf/2006.08651)
    '''

    Q11  = np.sum((dx**2)*w)/np.sum(w)
    Q22  = np.sum((dy**2)*w)/np.sum(w)
    Q12  = np.sum((dx*dy)*w)/np.sum(w)
    E1 = (Q11-Q22)/(Q11+Q22)
    E2 = (2.*Q12)/(Q11+Q22)
    e = np.sqrt(E1**2 + E2**2)
    theta = np.arctan2(E2,E1)/2.0
    return e,theta

def cluster_orientation(ra_cl, dec_cl, ra_mem, dec_mem, weight=None):
    if not weight:
        weight = np.ones_like(ra_mem)
    x, y = gnomonic_projection(ra_mem, dec_mem, ra_cl, dec_cl)
    e, theta = inertia(x, y, weight)
    return x, y, e, theta

def main():

    l = CLUSTERS[ (CLUSTERS['lambda']>LMIN) & (CLUSTERS['lambda']<=LMAX) & (CLUSTERS['redshift']>ZMIN) & (CLUSTERS['redshift']<=ZMAX) ]
    id_cl = l['mem_match_id']

    fig, ax = plt.subplots(1,1)    
    for idx in id_cl:
        m = MEMBERS[MEMBERS['mem_match_id']==idx]
        x, y, e, theta = cluster_orientation(l[l['mem_match_id']==idx]['ra_cl'], l[l['mem_match_id']==idx]['dec_cl'], m['ra_mem'], m['dec_mem'])

        print(f'{e=}')
        print(f'{theta=}')

        ax.scatter(x, y, s=10, edgecolor='b', facecolor='none')
        ax.axline((0.0,0.0), slope=np.tan(theta), c='k')
        #ax.axline((0.0,0.0), slope=np.tan(theta+np.pi), c='k')
    plt.show()

if __name__=='__main__':
    main()