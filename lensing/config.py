import numpy as np
import toml

# ==== Input globals
# read from config file
class Config:

    def __init__(self, configfile:str='lensing/config.toml'):

        config = toml.load(configfile)

        self.lensname=config['catalog']['lenses']
        self.sourcename=config['catalog']['sources']
        self.anglesname = config['catalog']['angles']
        
        self.sample=config['run']['sample']
        self.NCORES = config['run']['ncores']
        self.PLOT = config['run']['plot']
        self.OVERWRITE = config['run']['overwrite']
        
        self.RIN, self.ROUT = config['profile']['rin'], config['profile']['rout'] #Mpc/h
        self.NBINS = config['profile']['nbins']
        self.NJK = config['profile']['njk']
        self.BINNING = config['profile']['binning']

        self.PCEN = config['lensescut']['pcen']
        self.ZBINS = self._edges_to_bins(config['lensescut']['zedges'], 'ZEDGES')
        self.LBINS = self._edges_to_bins(config['lensescut']['ledges'], 'LEDGES')
        self.MEMCUT = config['lensescut']['memcut']
        self.ANGWEIGHT = config['lensescut']['angweight']


    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))



