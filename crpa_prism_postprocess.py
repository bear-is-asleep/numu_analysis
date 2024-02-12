#Boiler plate imports
import uproot 
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.interpolate import interp1d

import xml.etree.ElementTree as ET

#SBND imports
sys.path.append('/exp/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
from sbnd.general import utils,plotters
from sbnd.plotlibrary import makeplot
from sbnd import stats as sbndstats
from sbnd import prism
from sbnd.constants import *
from sbnd.cafclasses import object_calc

#Variables
save_plots = True #save plots
close_plots = False #close plots after saving
save_df = True #save postprocessed dataframes
#variables for setting and checking data
check_flux = False #check flux histograms
load_dfs = False #load postprocessed dataframes
make_dfs = True #make from genie trees

#Normalization constants
flux_pot_norm = 1e6 #Flux normalized to 1e6 pot
m2_per_cm2 = 1e-4 #m^2 per cm^2
n_events = 1e6 #events in each sample
ratio_mcp_to_g18 = [18610/0.2117749 #0th prism bin
  ,55342/0.6452452 #1st prism bin
  ,88414/1.0591616 #2nd prism bin
  ,102354/1.3367942 #3rd prism bin
  ,101847/1.4145875 #4th prism bin
  ,71133/1.1684591 #5th prism bin
  ]
#ratio_mcp_to_g18 *= 12 #scale to 3 years (comment out for 3 months)
#ratio of mcp to g18 events, measured by ratio in MCP plot and G18 with weighting
ratio_mcp_to_g18_mean = np.mean(ratio_mcp_to_g18)
ratio_mcp_to_g18_unc = np.std(ratio_mcp_to_g18)/ratio_mcp_to_g18_mean

#Constants
PLOTS_DIR = f'Plots/gevgencc_prism_{plotters.day}_full'
DATA_DIR = '/exp/sbnd/data/users/brindenc/genie/crpa/gevgens/numu/prism_processed/1m'

CRPA_LABEL = 'HF-CRPA (G21_11a)'
G21_LABEL = 'SuSAv2 (G21_11a)'
G18_LABEL = 'NAV (G18_10a_02_11a)'
HYBRID_LABEL = 'HF-CRPA SuSAv2 (G21_11a)'
MODEL_LABELS = [CRPA_LABEL,G21_LABEL,G18_LABEL,HYBRID_LABEL]
MODEL_COLORS = ['blue','orange','green','red']
MODEL_NAMES = ['crpa','susav2','nav','hybrid']

#Bins for plotting
COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
THETA_BINS = np.arccos(COSTHETA_BINS)

MOMENTUM_BINS = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,2,3])
PRISM_BINS = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
PRISM_RADII = [prism.calc_rf(theta) for theta in PRISM_BINS]
PRISM_AREAS = [prism.calc_prism_area(PRISM_RADII[i],PRISM_RADII[i+1]) for i in range(len(PRISM_RADII)-1)]
PRISM_LABELS = [r'$\theta_{PRISM} \in$' + f'[{PRISM_BINS[i]},{PRISM_BINS[i+1]}]' for i in range(len(PRISM_BINS)-1)]
#Make list of colors from colormap the same length as thetas
PRISM_COLORS = plotters.get_colors('gnuplot2',len(PRISM_BINS)-1)

SIM_LABEL = r'$\nu_\mu$ CC GENIE SBND'

#File names
fname_crpa_prism = '/exp/sbnd/data/users/brindenc/genie/crpa/gevgens/numu/CRPA/prism/1m/genie_tree_prism_binXX.root'
fname_hybrid_prism = '/exp/sbnd/data/users/brindenc/genie/crpa/gevgens/numu/CRPASuSAv2/prism/1m/genie_tree_prism_binXX.root'
fname_g18_prism = '/exp/sbnd/data/users/brindenc/genie/crpa/gevgens/numu/G18/prism/100k/genie_tree_prism_binXX.root' #this hasn't finished running yet
fname_g21_prism = '/exp/sbnd/data/users/brindenc/genie/crpa/gevgens/numu/G21/prism/1m/genie_tree_prism_binXX.root'
fname_sbnd_prism_flux = '/exp/sbnd/data/users/brindenc/genie/crpa/splines/prism_fluxes.root'
fname_sbnd_flux = '/exp/sbnd/data/users/brindenc/genie/crpa/splines/sbnd_flux.root'

#Lists of dataframes
dfs_crpa = []
dfs_hybrid = []
dfs_g18 = []
dfs_g21 = []
hists_flux = []

#Retrieve dataframes and histograms
flux_sbnd = uproot.open(fname_sbnd_flux)['flux_sbnd_numu;1']
for i in range(len(PRISM_BINS)-2):
  t0 = time()
  hist_sbnd_flux = uproot.open(fname_sbnd_prism_flux)[f'prism_flux_{i};1'] #make sure to use correct flux
  hists_flux.append(hist_sbnd_flux)
  if load_dfs: #this means we're going to directly load postprocessed dataframes
    dfs_crpa.append(pd.read_hdf(f'{DATA_DIR}/crpa.df',key=f'prism_bin{i}'))
    dfs_g18.append(pd.read_hdf(f'{DATA_DIR}/g18.df',key=f'prism_bin{i}'))
    dfs_g21.append(pd.read_hdf(f'{DATA_DIR}/g21.df',key=f'prism_bin{i}'))
    dfs_hybrid.append(pd.read_hdf(f'{DATA_DIR}/hybrid.df',key=f'prism_bin{i}'))
    print(f'Loaded dataframes for prism bin {i} from {DATA_DIR}')
    continue
  tree_crpa = uproot.open(fname_crpa_prism.replace('XX',str(i)))['genie_tree;1']
  dfs_crpa.append(tree_crpa.arrays(library='pd'))
  
  tree_hybrid = uproot.open(fname_hybrid_prism.replace('XX',str(i)))['genie_tree;1']
  dfs_hybrid.append(tree_hybrid.arrays(library='pd'))
  
  tree_g18 = uproot.open(fname_g18_prism.replace('XX',str(i)))['genie_tree;1']
  dfs_g18.append(tree_g18.arrays(library='pd'))
  
  tree_g21 = uproot.open(fname_g21_prism.replace('XX',str(i)))['genie_tree;1']
  dfs_g21.append(tree_g21.arrays(library='pd'))
  t1 = time()
  print(f'Loaded dataframes for prism bin {i} in {t1-t0:.2f} seconds')

Es = np.arange(0.01,5,0.1)
max_weight = 0
max_area = np.max(PRISM_AREAS)
hist_sbnd_fluxes_np = [None]*(len(PRISM_BINS)-2)

model_dfs = [dfs_crpa,dfs_hybrid,dfs_g18,dfs_g21]

for i in range(len(PRISM_BINS)-2):
  #Get prism bin area
  area = PRISM_AREAS[i]
  
  #Set up the flux
  values_sbnd_flux = hists_flux[i].values()*flux_pot_norm #m^-2
  edges_sbnd_flux = hists_flux[i].axis().edges()
  dE = edges_sbnd_flux[1] - edges_sbnd_flux[0] #this is needed for ratio plots [GeV]
  hist_sbnd_fluxes_np[i] = [values_sbnd_flux,edges_sbnd_flux] #get sbnd flux histograms into numpy histogram format
  
  centers_sbnd_flux = (edges_sbnd_flux[:-1] + edges_sbnd_flux[1:])/2
  integral = np.sum(values_sbnd_flux)*dE

  #Make flux spline
  flux_interp = interp1d(centers_sbnd_flux,values_sbnd_flux,kind='cubic',fill_value='extrapolate')
  if check_flux:
    
    #Check flux spline
    fig,ax = plt.subplots(figsize=(8,6))
    makeplot.plot_hist_edges(edges_sbnd_flux,values_sbnd_flux,None,r'$\nu_\mu$ Flux',ax=ax,lw=2)
    ax.plot(Es,flux_interp(Es),label='Interpolated',lw=2,linestyle='-.')
    ax.legend()
    plotters.set_style(ax)
    plotters.add_label(ax,f'mode = {sbndstats.calc_mean_hist(values_sbnd_flux,edges_sbnd_flux):.2f}'\
                        '\n'+f'mean = {sbndstats.calc_mode_hist(values_sbnd_flux,edges_sbnd_flux):.2f}'\
                        '\n'+f'integral = {integral:.2e}' 
                      ,where='bottomright'
                      ,fontsize=20)

    if save_plots:
      plotters.save_plot(f'flux_check_prism_bin{i}',fig=fig,folder_name=PLOTS_DIR)
    if close_plots:
      plt.close(fig)    
  if make_dfs:
    for j,df in enumerate(model_dfs):
      df = df[i]
      df['final_mu_p'] = np.linalg.norm(df[['final_mu_px','final_mu_py','final_mu_pz']].values,axis=1)
      df['final_mu_theta'] = np.arccos(df.final_mu_pz/df.final_mu_p)
      df['w'] = df.initialnu_E - df.final_mu_E
      df['costheta'] = np.cos(df.final_mu_theta)
      #N = sigma*flux*targets -> use proper units. The weight of each event can be interpreted as 
      #the ratio of the number of events generated to the number of events that would be observed in the detector
      df['weight'] = df['xsecs']*flux_interp(df['initialnu_E'])\
        *area/FACE_AREA*NUMBER_TARGETS/GeV2perm2
      #normalize to events generated when compared to MCP2022A G18 data 
      if j == 2: 
        df['event_weight'] = df['weight']*ratio_mcp_to_g18_mean 
      else:
        df['event_weight'] = df['weight']*ratio_mcp_to_g18_mean/10 #divide by 10 for now since G21 has 1m events, not 100k
      if np.max(df['weight']) > max_weight:
        max_weight = np.max(df['weight'])
      #Assign binnings for momentum and theta
      df = object_calc.get_df_from_bins(df,df,COSTHETA_BINS,'costheta','costheta_bin',ntuple_keys=False)
      df = object_calc.get_df_from_bins(df,df,MOMENTUM_BINS,'final_mu_p','momentum_bin',ntuple_keys=False)
      print(f'Finished binning for {MODEL_NAMES[j]} in prism bin {i}')
    if save_df:
      dfs_crpa[i].to_hdf(f'{DATA_DIR}/crpa.df',key=f'prism_bin{i}')
      dfs_hybrid[i].to_hdf(f'{DATA_DIR}/hybrid.df',key=f'prism_bin{i}')
      dfs_g18[i].to_hdf(f'{DATA_DIR}/g18.df',key=f'prism_bin{i}')
      dfs_g21[i].to_hdf(f'{DATA_DIR}/g21.df',key=f'prism_bin{i}')
      print(f'Saved dataframes for prism bin {i} to {DATA_DIR}')    

#SBND total flux
sbnd_flux = [np.zeros(len(hist_sbnd_fluxes_np[0][0])),None]
for i in range(len(PRISM_BINS)-2):
  sbnd_flux[0]+=hist_sbnd_fluxes_np[i][0] #values
sbnd_flux[1] = hist_sbnd_fluxes_np[0][1] #edges

dfs_list = [dfs_crpa,dfs_g21,dfs_g18,dfs_hybrid]
    
