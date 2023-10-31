#!/usr/bin/env python
# coding: utf-8

"""
Add variables and cuts to truth level objects for both genie and gibuu
"""

#Boiler plate imports
import pandas as pd
import sys
from time import time

#SBND imports
s0 = time()
sys.path.append('/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
from sbnd.general import utils
from sbnd.cafclasses.nu import NU
from sbnd.cafclasses.mcprim import MCPRIM

#Numu imports 
from sbnd.numu.numu_constants import *
s1 = time()
print(f'SBND imports: {s1-s0:.2f} s')

#Constants/variables
DATA_DIR  = '/sbnd/data/users/brindenc/analyze_sbnd/numu/'
process_genie = True
process_gibuu = False

if process_gibuu:
  print('+'*90)
  #Load gibuu df
  fname = '/sbnd/data/users/brindenc/analyze_sbnd/numu/gibuu/gibuu_v0.df'
  print(f'Processing GIBUU: {fname}')
  #Neutrino processing
  s0 = time()
  gibuu_nu_df = pd.read_hdf(fname, key='mcnu')
  gibuu_nu = NU(gibuu_nu_df)
  gibuu_nu.postprocess_and_cut() #inplace - add variables and cuts
  s1 = time()
  print(f'-gibuu nu time: {s1-s0:.2f} s')

  #Primary processing
  gibuu_prim_df = pd.read_hdf(fname, key='mcprim')
  gibuu_prim = MCPRIM(gibuu_prim_df
                      ,prism_bins=PRISM_BINS
                      ,momentum_bins=MOMENTUM_BINS
                      ,costheta_bins=COSTHETA_BINS)
  gibuu_prim.postprocess(nu=gibuu_nu) #adds binning to primaries, add values, apply nu cuts
  s2 = time()
  print(f'-gibuu prim time: {s2-s1:.2f} s')

  #Save to hdf5
  gibuu_nu.data.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcnu')
  gibuu_prim.data.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcprim')
  s3 = time()
  print(f'-gibuu save time: {s3-s2:.2f} s')
  print(f'-gibuu total time: {s3-s0:.2f} s')

if process_genie:
  print('+'*90)
  #Load genie df
  fname = '/sbnd/data/users/brindenc/analyze_sbnd/numu/MCP2022A/MCP2022A_10k.df'
  print(f'Processing GENIE: {fname}')

  #Neutrino processing
  s0 = time()
  genie_nu_df = pd.read_hdf(fname, key='mcnu')
  #genie_nu_df.loc[:,'genweight'] = np.ones(len(genie_nu_df)) #add genie weight to weight events to POT
  genie_nu = NU(genie_nu_df)
  genie_nu.postprocess_and_cut() #inplace - add variables and cuts
  s1 = time()
  print(f'-genie nu time: {s1-s0:.2f} s')

  #Primary processing
  genie_prim_df = pd.read_hdf(fname, key='mcprim')
  genie_prim = MCPRIM(genie_prim_df
                      ,prism_bins=PRISM_BINS
                      ,momentum_bins=MOMENTUM_BINS
                      ,costheta_bins=COSTHETA_BINS)
  genie_prim.postprocess(nu=genie_nu) #adds binning to primaries, add values, apply nu cuts
  s2 = time()
  print(f'-genie prim time: {s2-s1:.2f} s')

  #Save processed dfs to hdf5
  genie_nu.data.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcnu')
  genie_prim.data.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcprim')
  s3 = time()
  print(f'-genie save time: {s3-s2:.2f} s')
  print(f'-genie total time: {s3-s0:.2f} s')