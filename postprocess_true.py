#!/usr/bin/env python
# coding: utf-8

"""
Add variables and cuts to truth level objects for both genie and gibuu
"""

#Boiler plate imports
import uproot 
import pandas as pd
import sys
import numpy as np
from time import time

#SBND imports
sys.path.append('/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
from sbnd.general import utils
from sbnd.cafclasses.nu import NU
from sbnd.cafclasses.mcprim import MCPRIM

#Constants/variables
DATA_DIR  = '/sbnd/data/users/brindenc/analyze_sbnd/numu/'
process_genie = True
process_gibuu = True

if process_gibuu:
  print('+'*90)
  #Load gibuu df
  fname = '/sbnd/data/users/brindenc/analyze_sbnd/numu/gibuu/gibuu_v0.df'
  print(f'Processing GIBUU: {fname}')
  #Neutrino processing
  s0 = time()
  gibuu_nu_df = pd.read_hdf(fname, key='mcnu')
  gibuu_nu = NU(gibuu_nu_df)
  gibuu_nu.postprocess() #inplace - add variables
  gibuu_nu = gibuu_nu.all_cuts() #numucc cuts
  s1 = time()
  print(f'-gibuu nu time: {s1-s0:.2f} s')

  #Primary processing
  gibuu_prim_df = pd.read_hdf(fname, key='mcprim')
  gibuu_prim = MCPRIM(gibuu_prim_df)
  gibuu_prim = gibuu_prim.postprocess(nu=gibuu_nu)

  #Cut to numucc events for all primaries
  inds = utils.get_inds_from_sub_inds(set(gibuu_prim.index.values),set(gibuu_nu.index.values),3)
  gibuu_prim = MCPRIM(gibuu_prim.loc[inds])
  s2 = time()
  print(f'-gibuu prim time: {s2-s1:.2f} s')

  #Save to hdf5
  gibuu_nu.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcnu')
  gibuu_prim.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcprim')
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
  genie_nu.postprocess() #inplace - add variables
  genie_nu = genie_nu.all_cuts() #numucc cuts
  s1 = time()
  print(f'-genie nu time: {s1-s0:.2f} s')

  #Primary processing
  genie_prim_df = pd.read_hdf(fname, key='mcprim')
  genie_prim = MCPRIM(genie_prim_df)
  genie_prim = genie_prim.postprocess(nu=genie_nu)
  
  #Cut to numucc events for all primaries
  inds = utils.get_inds_from_sub_inds(set(genie_prim.index.values),set(genie_nu.index.values),3)
  genie_prim = MCPRIM(genie_prim.loc[inds])
  s2 = time()
  print(f'-genie prim time: {s2-s1:.2f} s')

  #Save processed dfs to hdf5
  genie_nu.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcnu')
  genie_prim.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcprim')
  s3 = time()
  print(f'-genie save time: {s3-s2:.2f} s')
  print(f'-genie total time: {s3-s0:.2f} s')