from astropy.table import Table

def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits'):
    return Table.read(filename)

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_25314.fits'):
    return Table.read(filename)


