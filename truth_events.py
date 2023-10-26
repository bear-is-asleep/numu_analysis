#!/usr/bin/env python
# coding: utf-8

#Boiler plate imports
import uproot 
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from time import time

#Other imports
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib.patches import Patch

#SBND imports
sys.path.append('/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
from makedf.branches import *
from sbnd.general import utils,plotters
from sbnd.volume import involume
from sbnd.numu import selection
from sbnd.cafclasses.nu import NU
from sbnd.cafclasses.mcprim import MCPRIM
from sbnd.constants import *
from sbnd.prism import *
from pyanalib import panda_helpers

#Constants/variables
PLOTS_DIR = f'Plots/nu_{plotters.day}'
DATA_DIR  = '/sbnd/data/users/brindenc/analyze_sbnd/numu/'
GENIE_MODEL = 'GENIE 3.0.6 G18_10a_02_11a'
GIBUU_MODEL = 'GiBUU 2021'
SIM_LABEL = 'SBND Simulation'

load_processed = True #load preprocessed dataframes
save_plots = True


# ## Prepare data (only needs to be done once)

if not load_processed:

  #Load gibuu df
  fname = '/sbnd/data/users/brindenc/analyze_sbnd/numu/gibuu/gibuu_v0.df'

  gibuu_nu_df   = pd.read_hdf(fname, key='mcnu')
  gibuu_nu = NU(gibuu_nu_df)

  gibuu_prim_df = pd.read_hdf(fname, key='mcprim')
  gibuu_prim = MCPRIM(gibuu_prim_df)

  gibuu_hdr_df = pd.read_hdf(fname, key='hdr')

  gibuu_nu.postprocess() #inplace
  gibuu_nu = gibuu_nu.all_cuts() #numucc cuts

  gibuu_prim = gibuu_prim.postprocess()
  
  #Cut to numucc events for all primaries
  inds = utils.get_inds_from_sub_inds(set(gibuu_prim.index.values),set(gibuu_nu.index.values),3)
  gibuu_prim = MCPRIM(gibuu_prim.loc[inds])

  #Load genie df
  fname = '/sbnd/data/users/brindenc/analyze_sbnd/numu/MCP2022A/MCP2022A_10k.df'

  genie_nu_df   = pd.read_hdf(fname, key='mcnu')
  genie_nu_df.loc[:,'genweight'] = np.ones(len(genie_nu_df))
  genie_nu = NU(genie_nu_df)

  genie_prim_df = pd.read_hdf(fname, key='mcprim')
  genie_prim = MCPRIM(genie_prim_df)

  genie_hdr_df = pd.read_hdf(fname, key='hdr')

  #Post processing
  genie_nu.postprocess() #inplace
  genie_nu = genie_nu.all_cuts() #numucc cuts

  genie_prim = genie_prim.postprocess()
  #Cut to numucc events for all primaries
  inds = utils.get_inds_from_sub_inds(set(genie_prim.index.values),set(genie_nu.index.values),3)
  genie_prim = MCPRIM(genie_prim.loc[inds])

  #Save processed dfs to hdf5
  genie_nu.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcnu')
  genie_prim.to_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcprim')

  gibuu_nu.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcnu')
  gibuu_prim.to_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcprim')


# ## Load Preprocessed data (do this if you've already processed the data)
if load_processed:
  genie_nu = NU(pd.read_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcnu'))
  genie_prim = MCPRIM(pd.read_hdf(f'{DATA_DIR}/MCP2022A/mcp2022A_processed.h5', key='mcprim'))
  genie_hdr_df = pd.read_hdf(f'{DATA_DIR}/MCP2022A/MCP2022A_10k.df', key='hdr')
  
  gibuu_nu = NU(pd.read_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcnu'))
  gibuu_prim = MCPRIM(pd.read_hdf(f'{DATA_DIR}/gibuu/gibuu_v0_processed.h5', key='mcprim'))
  gibuu_hdr_df = pd.read_hdf(f'{DATA_DIR}/gibuu/gibuu_v0.df', key='hdr')

GENIE_POT = np.sum(genie_hdr_df.pot)
GIBUU_POT = np.sum(gibuu_hdr_df.pot)

#Normalize to 10e20 POT / 12 -> 3 months
NOM_POT = 10e20/12 #3 months

#Scale genweight to POT
genie_nu.scale_to_pot(NOM_POT,GENIE_POT)
gibuu_nu.scale_to_pot(NOM_POT,GIBUU_POT)

#More constants
GIBUU_LABEL = f'{SIM_LABEL}\n{GIBUU_MODEL}\n{NOM_POT:.2e} POT'
GENIE_LABEL = f'{SIM_LABEL}\n{GENIE_MODEL}\n{NOM_POT:.2e} POT'


# ## PRISM Plots

#Prism bins
thetas = np.arange(0,1.8,0.2)
PRISM_BINS = thetas.copy()
# print(len(thetas))
# thetas = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.8])
# print(len(thetas))
# thetas = np.array([0,0.4,0.6,0.8,0.9,1,1.2,1.4,1.8])
# print(len(thetas))
# thetas = np.array([0,0.4,0.5,0.6,0.7,0.8,1,1.3,1.8])
# print(len(thetas))
# thetas = np.array([0,0.35,0.5,0.65,0.8,0.95,1.1,1.2,1.8])
# print(len(thetas))
# thetas = np.array([0,0.35,0.55,0.65,0.75,0.9,1.05,1.25,1.8])
# print(len(thetas))
# thetas = np.array([0,0.4,0.55,0.7,0.8,0.9,1.0,1.25,1.8])
# print(len(thetas))

def make_prism_rings(theta,ax,**pltkwargs):
  center = prism_centroid
  radius = calc_rf(theta)
  [ax.add_patch(Circle(center,radius=r1,**pltkwargs)) for r1 in radius]
  return ax
def make_prism_plot(mcnu,**pltkwargs):
  fig,ax = plt.subplots(figsize=(10,8))
  im = ax.hist2d(mcnu.position.x,mcnu.position.y,**pltkwargs)#,norm=colors.LogNorm())
  cbar = fig.colorbar(im[3],ax=ax)
  cbar.ax.tick_params(labelsize=16)
  
  #prism lines
  ax.scatter(-74,0,s=200,c='red',marker='x')
  ax = make_prism_rings(thetas,ax,fill=False,ls='--',lw=2,color='red',alpha=0.4)
  
  
  ax.set_xlabel('x [cm]')
  ax.set_ylabel('y [cm]')
  ax.set_title(rf'{round(np.sum(mcnu.genweight.values)):,} $\nu_\mu CC$ events')
  return fig,ax

#Get prism values
genie_prism_thetas = 180*genie_nu.loc[:,'theta']/np.pi
gibuu_prism_thetas = 180*gibuu_nu.loc[:,'theta']/np.pi


#Gibuu
fig,ax = make_prism_plot(gibuu_nu,bins=20,cmap='Blues',weights=gibuu_nu.genweight)

plotters.add_label(ax,GIBUU_LABEL,fontsize=20,alpha=0.8,color='darkred')
plotters.set_style(ax)
if save_plots:
  plotters.save_plot('gibuu_prism',fig=fig,folder_name=PLOTS_DIR)





#Genie
fig,ax = make_prism_plot(genie_nu,bins=20,cmap='Blues',weights=genie_nu.genweight)

plotters.add_label(ax,GENIE_LABEL,fontsize=20,alpha=0.8,color='darkred')
plotters.set_style(ax)
if save_plots:
  plotters.save_plot('genie_prism',fig=fig,folder_name=PLOTS_DIR)







colors = ['red','blue','green','orange','purple','cyan','magenta','yellow']
def make_mode_plots(mcnu,mode_map,weights=None,ylabel='Events',bins=np.arange(0,5.1,0.1),density=False,title=None,
                    ax=None,fig=None,**pltkwargs):
  norm = len(mcnu)/np.sum(mcnu.genweight.values)
  modes = np.unique(mcnu.genie_mode.values)
  Es = [None]*len(modes)
  counts = Es.copy()
  labels = Es.copy()
  if weights is not None:
    weight_modes = Es.copy()
  for i,mode in enumerate(modes):
    Es[i] = list(mcnu[mcnu.genie_mode==mode].E) #Get energy from mode
    labels[i] = f'{mode_map[mode]} : {round(len(Es[i])/norm):,}' #Mode label and 
    if weights is not None:
      weight_modes[i] = mcnu[mcnu.genie_mode==mode].genweight
  if fig is None and ax is None: #Make figure if not provided
    fig,ax = plt.subplots(figsize=(10,8))
  if not density:
    if weights is None:
      ax.hist(Es,stacked=True,label=labels,bins=bins,**pltkwargs)
    else:
      ax.hist(Es,stacked=True,label=labels,weights=weight_modes,bins=bins,**pltkwargs)
  if density:
    # Calculate total counts in each bin across all modes first
    if weights is None:
      total_counts, edges = np.histogram(np.concatenate(Es), bins=bins)
    else:
      total_counts, edges = np.histogram(np.concatenate(Es), bins=bins, weights=np.concatenate(weight_modes))
    bottom = np.zeros(len(total_counts))
    actual_counts = bottom.copy()
    for i,_ in enumerate(modes):
      if weights is None:
        counts, edges = np.histogram(Es[i], bins=bins)
      else:
        counts, edges = np.histogram(Es[i], weights=weight_modes[i], bins=bins)
      actual_counts += counts
      fractions = counts / total_counts
      ax.bar(edges[:-1], height=fractions, bottom=bottom, align='edge', width=np.diff(edges), label=labels[i],fill=True, **pltkwargs)
      # Add the fraction as text in the middle of the bar
      # bar_centers = edges[:-1] + np.diff(edges) / 2  # Calculate the center of each bar
      # for i,(center, fraction) in enumerate(zip(bar_centers, fractions)):
      #   if np.isnan(fraction): continue
      #   ax.text(center, bottom[i] + fraction / 2, f'{fraction*100:.1f}%', ha='center', va='center',rotation=90)
      bottom += fractions  # Update the bottom for the next mode
      bottom = [b if not np.isnan(b) else 0 for b in bottom]
    ax.grid(True)
  if title is None:
    title = rf'{round(len(mcnu)/norm):,} $\nu_\mu CC$ events'
  ax.set_title(title)
  ax.set_xlabel(r'$E_\nu$ [GeV]')
  if ylabel is not None:
    ax.set_ylabel(f'{ylabel} / {round((bins[1]-bins[0])*1e3):,} MeV')
  return fig,ax





#GENIE
for i,dens in enumerate(['','_dens']):
  fig,ax = make_mode_plots(genie_nu,GENIE_INTERACTION_MAP,weights=genie_nu.genweight,
                          density=True if i == 1 else False, #set density
                          ylabel='Normalized events' if i == 1 else 'Events',
                          )
  ax.legend()
  plotters.set_style(ax)
  if i == 0:
    plotters.add_label(ax,GENIE_LABEL,where='bottomright',fontsize=20,alpha=0.8,color='darkred')
  elif i == 1:
    plotters.add_label(ax,GENIE_LABEL,where='topright',fontsize=20,alpha=0.9,color='black')
    ax.set_ylim([0,1])
    ax.set_xlim([0,5])
  if save_plots:
    plotters.save_plot(f'genie_modes{dens}',fig=fig,folder_name=PLOTS_DIR)

temp_gibuu_nu = NU(gibuu_nu.copy())
temp_gibuu_nu.genie_mode = temp_gibuu_nu.genie_mode.where((temp_gibuu_nu.genie_mode < 3) | (temp_gibuu_nu.genie_mode >= 32),2)

#GIBUU
for i,dens in enumerate(['','_dens']):
  fig,ax = make_mode_plots(temp_gibuu_nu,GIBUU_INTERACTION_MAP,weights=temp_gibuu_nu.genweight,
                          density=True if i == 1 else False, #set density
                          ylabel='Normalized events' if i == 1 else 'Events',
                          )
  ax.legend()
  plotters.set_style(ax)
  if i == 0:
    plotters.add_label(ax,GIBUU_LABEL,where='bottomright',fontsize=20,alpha=0.8,color='darkred')
  elif i == 1:
    plotters.add_label(ax,GIBUU_LABEL,where='topright',fontsize=20,alpha=0.9,color='black')
    ax.set_ylim([0,1])
    ax.set_xlim([0,5])
  if save_plots:
    plotters.save_plot(f'gibuu_modes{dens}',fig=fig,folder_name=PLOTS_DIR)

#Make these plots for different PRISM bins
fig_gibuu_all,axs_gibuu = plt.subplots(4,2,figsize=(10,20),sharex=True)
fig_gibuu_all_dens,axs_gibuu_dens = plt.subplots(4,2,figsize=(10,20),sharex=True)

fig_genie_all,axs_genie = plt.subplots(4,2,figsize=(10,20),sharex=True)
fig_genie_all_dens,axs_genie_dens = plt.subplots(4,2,figsize=(10,20),sharex=True)

for i,(ax_gibuu,ax_gibuu_dens,ax_genie,ax_genie_dens) in enumerate(zip(axs_gibuu.flatten(),axs_gibuu_dens.flatten(),axs_genie.flatten(),axs_genie_dens.flatten())):
  if thetas[i] == thetas[-1]: break #skip last bin to avoid range errors
  #Get neutrinos within theta bins
  genie_inds_inrange = genie_prism_thetas[(genie_prism_thetas <= thetas[i+1]) & (genie_prism_thetas > thetas[i])].index.values
  gibuu_inds_inrange = gibuu_prism_thetas[(gibuu_prism_thetas <= thetas[i+1]) & (gibuu_prism_thetas > thetas[i])].index.values
  
  #Get neutrino objects with these inds
  genie_nu_inrange = genie_nu.loc[genie_inds_inrange]
  temp_gibuu_nu_inrange = temp_gibuu_nu.loc[gibuu_inds_inrange]
  
  title = r'$\theta_{PRISM}' + rf' \in [{thetas[i]:.2f},{thetas[i+1]:.2f}]$'
  
  #Make plots
  for j,dens in enumerate(['','_dens']):
    if i %4 == 0: #is at the edge of a row
      ylabel='Normalized events' if j == 1 else 'Events'
    else:
      ylabel=None
    #GIBUU
    make_mode_plots(temp_gibuu_nu_inrange,GIBUU_INTERACTION_MAP,weights=temp_gibuu_nu_inrange.genweight,
                            density=True if j == 1 else False, #set density
                            ylabel=ylabel,
                            title=title + f' ({round(temp_gibuu_nu_inrange.genweight.sum()):,}' + r' $\nu_\mu CC$)',
                            ax=ax_gibuu if j == 0 else ax_gibuu_dens,
                            fig=fig_gibuu_all if j == 0 else fig_gibuu_all_dens
                            )
    
    #GENIE
    make_mode_plots(genie_nu_inrange,GENIE_INTERACTION_MAP,weights=genie_nu_inrange.genweight,
                            density=True if j == 1 else False, #set density
                            ylabel=ylabel,
                            title=title + f' ({round(genie_nu_inrange.genweight.sum()):,}' + r' $\nu_\mu CC$)',
                            ax=ax_genie if j == 0 else ax_genie_dens,
                            fig=fig_genie_all if j == 0 else fig_genie_all_dens,
                            )
    for k,ax in enumerate([ax_genie,ax_genie_dens,ax_gibuu,ax_gibuu_dens]):
      if k == 0 or k == 2: #legend for non density plots
        ax.legend()
      if i < 6:
        ax.set_xlabel(None) #Turn off xlabel for upper plots
      if k == 0 or k == 1:
        label = GENIE_MODEL
      else:
        label = GIBUU_MODEL
      plotters.add_label(ax,label,where='bottomright',fontsize=16,alpha=0.8,color='black')
      plotters.set_style(ax,legend_size=10,title_size=15)
      
if save_plots:
  plotters.save_plot(f'gibuu_modes_prism_dens',fig=fig_gibuu_all_dens,folder_name=PLOTS_DIR) 
  plotters.save_plot(f'gibuu_modes_prism',fig=fig_gibuu_all,folder_name=PLOTS_DIR) 
  plotters.save_plot(f'genie_modes_prism_dens',fig=fig_genie_all_dens,folder_name=PLOTS_DIR) 
  plotters.save_plot(f'genie_modes_prism',fig=fig_genie_all,folder_name=PLOTS_DIR) 


# ## Muon info

#Get muons
gibuu_muons = gibuu_prim.get_true_parts_from_pdg(13)
genie_muons = genie_prim.get_true_parts_from_pdg(13)

#Get angles
gibuu_costheta = np.cos(gibuu_muons.theta)*np.sign(gibuu_muons.theta)
genie_costheta = np.cos(genie_muons.theta)*np.sign(genie_muons.theta)

#Bins for plotting
costheta_bins = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
theta_bins = np.arccos(costheta_bins)

p_bins = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,2,3])

#indeces for setting weights
gibuu_inds = utils.get_inds_from_sub_inds(set(gibuu_costheta.index.values),set(gibuu_nu.index.values),3)
genie_inds = utils.get_inds_from_sub_inds(set(genie_costheta.index.values),set(genie_nu.index.values),3)

#Get weights
genie_weights = genie_nu.loc[genie_inds,'genweight'].values
gibuu_weights = gibuu_nu.loc[gibuu_inds,'genweight'].values

#Get number of muons from weights
genie_muon_count = np.sum(genie_weights)
gibuu_muon_count = np.sum(gibuu_weights)

#Get neutrino angle values for prism
genie_prism_thetas = 180*genie_nu.loc[genie_inds,'theta']/np.pi
gibuu_prism_thetas = 180*gibuu_nu.loc[gibuu_inds,'theta']/np.pi

#Get list of muons by prism bin
ps_gibuu_prismbins = [None]*len()

for i,dens in enumerate(['','_dens']):
    fig,ax = plt.subplots(figsize=(8,6))
    h = ax.hist([genie_costheta,gibuu_costheta],
        bins=np.arange(-1,1.1,0.1),
        weights=[genie_weights,gibuu_weights],
        histtype='step',
        lw=3,
        alpha=0.9,
        label=[f'{GENIE_MODEL} ({round(genie_muon_count):,})',
        f'{GIBUU_MODEL} ({round(gibuu_muon_count):,})',],
        density=True if i == 1 else False, #set density
        )
    ax.legend()
    ax.set_xlabel(r'$\cos\theta_\mu$')
    ax.set_ylabel('Normalized events' if i == 1 else 'Events')

    plotters.set_style(ax,legend_loc='upper left')
    plotters.add_label(ax,SIM_LABEL,alpha=0.7,fontsize=20,color='darkred',where='center')
    if save_plots:
        plotters.save_plot(f'costheta_mu{dens}',fig=fig,folder_name=PLOTS_DIR)

for i,dens in enumerate(['','_dens']):
    fig,ax = plt.subplots(figsize=(8,6))
    h = ax.hist([np.linalg.norm(genie_muons.genp,axis=1),
        np.linalg.norm(gibuu_muons.genp,axis=1),],
        bins=np.arange(0,4,0.1),
        weights=[genie_weights,gibuu_weights],
        histtype='step',
        lw=3,
        alpha=0.9,
        label=[f'{GENIE_MODEL} ({round(genie_muon_count):,})',
        f'{GIBUU_MODEL} ({round(gibuu_muon_count):,})',],
        density=True if i == 1 else False, #set density
        )
    ax.legend()
    ax.set_xlabel(r'$p_\mu$ [GeV]')
    ax.set_ylabel('Normalized events' if i == 1 else 'Events')

    plotters.set_style(ax,legend_loc='upper right')
    plotters.add_label(ax,SIM_LABEL,alpha=0.7,fontsize=20,color='darkred',where='center')
    if save_plots:
        plotters.save_plot(f'momentum_mu{dens}',fig=fig,folder_name=PLOTS_DIR)

#Make list of colors from colormap the same length as thetas
colors = plotters.get_colors('gnuplot2',len(thetas)-1)

#Prism plot for muon kinematics - let's do this all at once to avoid double counting errors
for i,dens in enumerate(['','_dens']):
    #Figures forr angles
    fig_gibuu_costheta,ax_gibuu_costheta = plt.subplots(figsize=(8,6))
    fig_genie_costheta,ax_genie_costheta = plt.subplots(figsize=(8,6))
    
    #Figures for momenta
    fig_gibuu_momentum,ax_gibuu_momentum = plt.subplots(figsize=(8,6))
    fig_genie_momentum,ax_genie_momentum = plt.subplots(figsize=(8,6))
    
    #Make a list for repeated tasks
    axs = [ax_gibuu_costheta,ax_genie_costheta,ax_gibuu_momentum,ax_genie_momentum]
    ax_thetas = [ax_gibuu_costheta,ax_genie_costheta]
    ax_momenta = [ax_gibuu_momentum,ax_genie_momentum]
    
    #Labels for figures
    labels_gibuu = [None]*(len(thetas)-1)
    labels_genie = labels_gibuu.copy()
    
    #Prism momenta
    ps_gibuu_prism = labels_gibuu.copy() #One histogram per theta bin
    ps_genie_prism = labels_gibuu.copy() #One histogram per theta bin
    
    #Prism angles
    costhetas_gibuu_prism = ps_gibuu_prism.copy()
    costhetas_genie_prism = ps_genie_prism.copy()
    
    #Prism weights
    weights_gibuu_prism = ps_gibuu_prism.copy()
    weights_genie_prism = ps_genie_prism.copy()
    
    for j,_ in enumerate(thetas):
        if thetas[j] == thetas[-1]: break #skip last bin to avoid range errors
        #Get neutrinos within theta bins
        genie_inds_inrange = genie_prism_thetas[(genie_prism_thetas <= thetas[j+1]) & (genie_prism_thetas > thetas[j])].index.values
        gibuu_inds_inrange = gibuu_prism_thetas[(gibuu_prism_thetas <= thetas[j+1]) & (gibuu_prism_thetas > thetas[j])].index.values
        
        #Convert to muon indeces
        genie_muon_inds_inrange = utils.get_inds_from_sub_inds(set(genie_muons.index.values),set(genie_inds_inrange),3)
        gibuu_muon_inds_inrange = utils.get_inds_from_sub_inds(set(gibuu_muons.index.values),set(gibuu_inds_inrange),3)
                
        #Get muons within theta bins
        genie_muons_inrange = genie_muons.loc[genie_muon_inds_inrange]
        gibuu_muons_inrange = gibuu_muons.loc[gibuu_muon_inds_inrange]        
        
        #Get momenta
        ps_genie_prism[j] = np.linalg.norm(genie_muons_inrange.genp,axis=1)
        ps_gibuu_prism[j] = np.linalg.norm(gibuu_muons_inrange.genp,axis=1)
        
        #Get costheta angles
        costhetas_genie_prism[j] = np.cos(genie_muons_inrange.theta.values)*np.sign(genie_muons_inrange.theta.values)
        costhetas_gibuu_prism[j] = np.cos(gibuu_muons_inrange.theta.values)*np.sign(gibuu_muons_inrange.theta.values)
        
        #Get weights
        weights_genie_prism[j] = genie_nu.loc[genie_inds_inrange,'genweight'].values
        weights_gibuu_prism[j] = gibuu_nu.loc[gibuu_inds_inrange,'genweight'].values
        
        #Set labels
        labels_genie[j] = f'{round(thetas[j],2)} < ' + r'$\theta_{PRISM}$' + f' < {round(thetas[j+1],2)} ({round(np.sum(weights_genie_prism[j])):,})'
        labels_gibuu[j] = f'{round(thetas[j],2)} < ' + r'$\theta_{PRISM}$' + f' < {round(thetas[j+1],2)} ({round(np.sum(weights_gibuu_prism[j])):,})'
    
    #Make histograms
    for k,data in enumerate(costhetas_gibuu_prism):
        ax_gibuu_costheta.hist(data, #cos theta values
            bins=np.arange(-1,1.1,0.1),
            weights=weights_gibuu_prism[k],
            histtype='step',
            lw=2,
            alpha=0.9,
            label=labels_gibuu[k],
            density=True if i == 1 else False, #set density
            linestyle='-' if k % 2 == 0 else '--', #alternate linestyle to help with visibility
            color=colors[k],
            )
    
    for k,data in enumerate(costhetas_genie_prism):
        ax_genie_costheta.hist(data, #cos theta values
            bins=np.arange(-1,1.1,0.1),
            weights=weights_genie_prism[k],
            histtype='step',
            lw=2,
            alpha=0.9,
            label=labels_genie[k],
            density=True if i == 1 else False, #set density
            color=colors[k],
            linestyle='-' if k % 2 == 0 else '--', #alternate linestyle to help with visibility
            )
    for k,data in enumerate(ps_gibuu_prism):
        ax_gibuu_momentum.hist(data, #momentum values
            bins=np.arange(0,4,0.1),
            weights=weights_gibuu_prism[k],
            histtype='step',
            lw=2,
            alpha=0.9,
            label=labels_gibuu[k],
            density=True if i == 1 else False, #set density
            color=colors[k],
            linestyle='-' if k % 2 == 0 else '--', #alternate linestyle to help with visibility
            )
    
    for k,data in enumerate(ps_genie_prism):
        ax_genie_momentum.hist(data, #cos theta values
            bins=np.arange(0,4,0.1),
            weights=weights_genie_prism[k],
            histtype='step',
            lw=2,
            alpha=0.9,
            label=labels_genie[k],
            density=True if i == 1 else False, #set density
            color=colors[k],
            linestyle='-' if k % 2 == 0 else '--', #alternate linestyle to help with visibility
            )
    
    #Set xlabels
    ax_gibuu_costheta.set_xlabel(r'$\cos\theta_\mu$')
    ax_gibuu_momentum.set_xlabel(r'$p_\mu$ [GeV]')
    ax_genie_costheta.set_xlabel(r'$\cos\theta_\mu$')
    ax_genie_momentum.set_xlabel(r'$p_\mu$ [GeV]')
    
    #Set labels
    plotters.add_label(ax_gibuu_costheta,GIBUU_LABEL,alpha=0.8,fontsize=18,color='darkred',where='bottomishleft')
    plotters.add_label(ax_genie_costheta,GENIE_LABEL,alpha=0.8,fontsize=18,color='darkred',where='bottomishleft')
    plotters.add_label(ax_gibuu_momentum,GIBUU_LABEL,alpha=0.8,fontsize=18,color='darkred',where='bottomishright')
    plotters.add_label(ax_genie_momentum,GENIE_LABEL,alpha=0.8,fontsize=18,color='darkred',where='bottomishright')
    
    for k,ax in enumerate(axs):
        ax.set_ylabel('Normalized events' if i == 1 else 'Events')
        ax.legend()
    for k,ax in enumerate(ax_thetas):
        plotters.set_style(ax,legend_size=11,legend_loc='upper left')
    for k,ax in enumerate(ax_momenta):
        plotters.set_style(ax,legend_size=11,legend_loc='upper right')
    if save_plots:
        plotters.save_plot(f'genie_prism_muon_costheta{dens}',fig=fig_genie_costheta,folder_name=PLOTS_DIR)
        plotters.save_plot(f'gibuu_prism_muon_costheta{dens}',fig=fig_gibuu_costheta,folder_name=PLOTS_DIR)
        plotters.save_plot(f'genie_prism_muon_momentum{dens}',fig=fig_genie_momentum,folder_name=PLOTS_DIR)
        plotters.save_plot(f'gibuu_prism_muon_momentum{dens}',fig=fig_gibuu_momentum,folder_name=PLOTS_DIR)






#fig and ax for 2d hists
fig_genie,ax_genie = plt.subplots(figsize=(12,10))
fig_gibuu,ax_gibuu = plt.subplots(figsize=(12,10))

#pie charts
fig_genie_pie, axs_genie_pi = plt.subplots(len(thetas)-1, len(costheta_bins)-1, figsize=(24,20))
#fig_gibuu_pie, axs_gibuu_pi = plt.subplots(len(thetas)-1, len(costheta_bins)-1, figsize=(12,10))

#Make legend labels for pie chart
colors_pie = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#9467bd']
labels_pie = ['QE','Res','DIS','MEC']
modes_genie = [0,1,2,10]
legend_elements = [Patch(facecolor=color, edgecolor=color, label=label) for label, color in zip(labels_pie, colors_pie)]

genie_angle_events = np.zeros((len(thetas)-1,len(costheta_bins)-1))
gibuu_angle_events = genie_angle_events.copy()

for i in range(len(thetas)-1): #Prism theta bins
  #if i >0: break
  #Get neutrinos within theta bins
  genie_inds_inrange = genie_prism_thetas[(genie_prism_thetas <= thetas[i+1]) & (genie_prism_thetas > thetas[i])].index.values
  gibuu_inds_inrange = gibuu_prism_thetas[(gibuu_prism_thetas <= thetas[i+1]) & (gibuu_prism_thetas > thetas[i])].index.values
  
  #Convert to muon indeces
  genie_muon_inds_inrange = utils.get_inds_from_sub_inds(set(genie_muons.index.values),set(genie_inds_inrange),3)
  gibuu_muon_inds_inrange = utils.get_inds_from_sub_inds(set(gibuu_muons.index.values),set(gibuu_inds_inrange),3)
          
  #Get muons within theta bins
  genie_muons_inrange = genie_muons.loc[genie_muon_inds_inrange]
  gibuu_muons_inrange = gibuu_muons.loc[gibuu_muon_inds_inrange]
  
  for j in range(len(theta_bins)-1): #Muon angle bins
    #if j < 8: continue
    #Get muons within theta bins
    genie_muons_inrange_intheta = genie_muons_inrange[(genie_muons_inrange.theta < theta_bins[j]) & (genie_muons_inrange.theta > theta_bins[j+1])]
    gibuu_muons_inrange_intheta = gibuu_muons_inrange[(gibuu_muons_inrange.theta < theta_bins[j]) & (gibuu_muons_inrange.theta > theta_bins[j+1])]
    
    #Get weights
    genie_inrange_intheta_inds = utils.get_inds_from_sub_inds(set(genie_muons_inrange_intheta.index.values),set(genie_nu.index.values),3)
    weights_genie = genie_nu.loc[genie_inrange_intheta_inds,'genweight'].values
    
    gibuu_inrange_intheta_inds = utils.get_inds_from_sub_inds(set(gibuu_muons_inrange_intheta.index.values),set(gibuu_nu.index.values),3)
    weights_gibuu = gibuu_nu.loc[gibuu_inrange_intheta_inds,'genweight'].values
    
    #Get number of events from weights
    genie_angle_events[i][j] = np.sum(weights_genie)
    gibuu_angle_events[i][j] = np.sum(weights_gibuu)
    
    #Get modes from indices
    genie_nus_inrange_intheta = genie_nu.loc[genie_inrange_intheta_inds]
    
    #Pie chart with different modes
    genie_mode_ratios = [np.sum(weights_genie[genie_nus_inrange_intheta.genie_mode == mode]) for mode in modes_genie]
    #genie_mode_ratios.remove(0) #remove 0s
    axs_genie_pi[i,j].pie(genie_mode_ratios,
                  colors=colors_pie,
                 #labels=GENIE_INTERACTION_MAP.values(), 
                 autopct='%1.0f%%'
                 )
    axs_genie_pi[i,j].set_xticks([])
    axs_genie_pi[i,j].set_yticks([])
    #plotters.set_style(axs_genie_pi[i,j])

#Prep tick labels
costheta_ticklabels = [f'{costheta_bins[i]:.2f} - {costheta_bins[i+1]:.2f}' for i in range(len(costheta_bins)-1)]
prismtheta_ticklabels = [f'{thetas[i]:.2f} - {thetas[i+1]:.2f}' for i in range(len(thetas)-1)]

#Genie
genie_im = ax_genie.imshow(genie_angle_events,cmap='Oranges')
for j in range(len(costheta_bins)-1):
    for i in range(len(thetas)-1):
        text = ax_genie.text(j,i, f'{round(genie_angle_events[i, j]):,}'+'\n'+f'{round(100*genie_angle_events[i, j]/np.sum(genie_angle_events),2):.2f}%', 
                             ha="center", va="center",fontsize=14)
ax_genie.set_title(SIM_LABEL+' '+GENIE_MODEL+f' ({round(genie_muon_count):,})')

#Gibuu
gibuu_im = ax_gibuu.imshow(gibuu_angle_events,cmap='Oranges')
for j in range(len(costheta_bins)-1):
    for i in range(len(thetas)-1):
        text = ax_gibuu.text(j, i, f'{round(gibuu_angle_events[i, j]):,}'+'\n'+f'{round(100*gibuu_angle_events[i, j]/np.sum(gibuu_angle_events),2):.2f}%', 
                             ha="center", va="center",fontsize=14)
ax_gibuu.set_title(SIM_LABEL+' '+GIBUU_MODEL+f' ({round(gibuu_muon_count):,})')

#Both
for ax in [ax_genie,ax_gibuu]:
  ax.set_xlabel(r'$\cos\theta_\mu$')
  ax.set_ylabel(r'$\theta_{PRISM}$')
  ax.set_xticks(np.arange(0,len(costheta_bins)-1))
  ax.set_yticks(np.arange(0,len(thetas)-1))
  ax.set_xticklabels(labels=costheta_ticklabels,rotation=30)
  ax.set_yticklabels(labels=prismtheta_ticklabels,rotation=30)
  plotters.set_style(ax)

#Pie plot
axs_genie_pi[0,-1].legend(handles=legend_elements,fontsize=18,bbox_to_anchor=(1.01,1))
plotters.add_label(axs_genie_pi[-1,-1],GENIE_LABEL,where='bottomrightoutside',fontsize=20,alpha=0.8,color='darkred')

if save_plots:
    plotters.save_plot('genie_prism_muon_costheta_counts',fig=fig_genie,folder_name=PLOTS_DIR)
    plotters.save_plot('gibuu_prism_muon_costheta_counts',fig=fig_gibuu,folder_name=PLOTS_DIR)
    plotters.save_plot('genie_prism_muon_costheta_pie',fig=fig_genie_pie,folder_name=PLOTS_DIR)





#I'm lazy so im copy pasting the above code to use for the momenta bins

#fig and ax for 2d hists
fig_genie,ax_genie = plt.subplots(figsize=(12,10))
fig_gibuu,ax_gibuu = plt.subplots(figsize=(12,10))

#pie charts
fig_genie_pie, axs_genie_pi = plt.subplots(len(thetas)-1, len(p_bins)-1, figsize=(24,20))
#fig_gibuu_pie, axs_gibuu_pi = plt.subplots(len(thetas)-1, len(costheta_bins)-1, figsize=(12,10))

#Make legend labels for pie chart
colors_pie = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#9467bd']
labels_pie = ['QE','Res','DIS','MEC']
modes_genie = [0,1,2,10]
legend_elements = [Patch(facecolor=color, edgecolor=color, label=label) for label, color in zip(labels_pie, colors_pie)]

genie_angle_events = np.zeros((len(thetas)-1,len(p_bins)-1))
gibuu_angle_events = genie_angle_events.copy()

for i in range(len(thetas)-1): #Prism theta bins
  #if i >0: break
  #Get neutrinos within theta bins
  genie_inds_inrange = genie_prism_thetas[(genie_prism_thetas <= thetas[i+1]) & (genie_prism_thetas > thetas[i])].index.values
  gibuu_inds_inrange = gibuu_prism_thetas[(gibuu_prism_thetas <= thetas[i+1]) & (gibuu_prism_thetas > thetas[i])].index.values
  
  #Convert to muon indeces
  genie_muon_inds_inrange = utils.get_inds_from_sub_inds(set(genie_muons.index.values),set(genie_inds_inrange),3)
  gibuu_muon_inds_inrange = utils.get_inds_from_sub_inds(set(gibuu_muons.index.values),set(gibuu_inds_inrange),3)
          
  #Get muons within theta bins
  genie_muons_inrange = genie_muons.loc[genie_muon_inds_inrange]
  gibuu_muons_inrange = gibuu_muons.loc[gibuu_muon_inds_inrange]
  
  for j in range(len(p_bins)-1): #Muon angle bins
    #if j < 8: continue
    #Extrac muon momenta
    genie_momentum_inrange = np.linalg.norm(genie_muons_inrange.genp,axis=1)
    gibuu_momentum_inrange = np.linalg.norm(gibuu_muons_inrange.genp,axis=1)
    
    #Get muons within momentum bins
    genie_muons_inrange_inp = genie_muons_inrange[(genie_momentum_inrange > p_bins[j]) & (genie_momentum_inrange < p_bins[j+1])]
    gibuu_muons_inrange_inp = gibuu_muons_inrange[(gibuu_momentum_inrange > p_bins[j]) & (gibuu_momentum_inrange < p_bins[j+1])]
    
    #Get weights
    genie_inrange_inp_inds = utils.get_inds_from_sub_inds(set(genie_muons_inrange_inp.index.values),set(genie_nu.index.values),3)
    weights_genie = genie_nu.loc[genie_inrange_inp_inds,'genweight'].values
    
    gibuu_inrange_inp_inds = utils.get_inds_from_sub_inds(set(gibuu_muons_inrange_inp.index.values),set(gibuu_nu.index.values),3)
    weights_gibuu = gibuu_nu.loc[gibuu_inrange_inp_inds,'genweight'].values
    
    #Get number of events from weights
    genie_angle_events[i][j] = np.sum(weights_genie)
    gibuu_angle_events[i][j] = np.sum(weights_gibuu)
    
    #Get modes from indices
    genie_nus_inrange_inp = genie_nu.loc[genie_inrange_inp_inds]
    
    #Pie chart with different modes
    genie_mode_ratios = [np.sum(weights_genie[genie_nus_inrange_inp.genie_mode == mode]) for mode in modes_genie]
    #genie_mode_ratios.remove(0) #remove 0s
    axs_genie_pi[i,j].pie(genie_mode_ratios,
                  colors=colors_pie,
                 #labels=GENIE_INTERACTION_MAP.values(), 
                 autopct='%1.0f%%'
                 )
    axs_genie_pi[i,j].set_xticks([])
    axs_genie_pi[i,j].set_yticks([])
    #plotters.set_style(axs_genie_pi[i,j])

#Prep tick labels
p_ticklabels = [f'{p_bins[i]:.2f} - {p_bins[i+1]:.2f}' for i in range(len(p_bins)-1)]
prismtheta_ticklabels = [f'{thetas[i]:.2f} - {thetas[i+1]:.2f}' for i in range(len(thetas)-1)]

#Genie
genie_im = ax_genie.imshow(genie_angle_events,cmap='Oranges')
for j in range(len(p_bins)-1):
    for i in range(len(thetas)-1):
        text = ax_genie.text(j,i, f'{round(genie_angle_events[i, j]):,}'+'\n'+f'{round(100*genie_angle_events[i, j]/np.sum(genie_angle_events),2):.2f}%', 
                             ha="center", va="center",fontsize=14)
ax_genie.set_title(SIM_LABEL+' '+GENIE_MODEL+f' ({round(genie_muon_count):,})')

#Gibuu
gibuu_im = ax_gibuu.imshow(gibuu_angle_events,cmap='Oranges')
for j in range(len(p_bins)-1):
    for i in range(len(thetas)-1):
        text = ax_gibuu.text(j, i, f'{round(gibuu_angle_events[i, j]):,}'+'\n'+f'{round(100*gibuu_angle_events[i, j]/np.sum(gibuu_angle_events),2):.2f}%', 
                             ha="center", va="center",fontsize=14)
ax_gibuu.set_title(SIM_LABEL+' '+GIBUU_MODEL+f' ({round(gibuu_muon_count):,})')

#Both
for ax in [ax_genie,ax_gibuu]:
  ax.set_xlabel(r'$p_\mu$ [GeV]')
  ax.set_ylabel(r'$\theta_{PRISM}$')
  ax.set_xticks(np.arange(0,len(p_bins)-1))
  ax.set_yticks(np.arange(0,len(thetas)-1))
  ax.set_xticklabels(labels=p_ticklabels,rotation=30)
  ax.set_yticklabels(labels=prismtheta_ticklabels,rotation=30)
  plotters.set_style(ax)

#Pie plot
axs_genie_pi[0,-1].legend(handles=legend_elements,fontsize=18,bbox_to_anchor=(1.01,1))
plotters.add_label(axs_genie_pi[-1,-1],GENIE_LABEL,where='bottomrightoutside',fontsize=20,alpha=0.8,color='darkred')

if save_plots:
    plotters.save_plot('genie_prism_muon_momentum_counts',fig=fig_genie,folder_name=PLOTS_DIR)
    plotters.save_plot('gibuu_prism_muon_momentum_counts',fig=fig_gibuu,folder_name=PLOTS_DIR)
    plotters.save_plot('genie_prism_muon_momentum_pie',fig=fig_genie_pie,folder_name=PLOTS_DIR)





for j,dens in enumerate(['','_dens']):
  fig,axs = plt.subplots(nrows=3,ncols=3,figsize=(15,12),sharex=True)
  for i,ax in enumerate(axs.flatten()):
    #Get muons within theta bins
    genie_muons_inrange = genie_muons[(genie_muons.theta < theta_bins[i]) & (genie_muons.theta > theta_bins[i+1])]
    gibuu_muons_inrange = gibuu_muons[(gibuu_muons.theta < theta_bins[i]) & (gibuu_muons.theta > theta_bins[i+1])]
    
    #Get momenta
    ps_genie = np.linalg.norm(genie_muons_inrange.genp,axis=1)
    ps_gibuu = np.linalg.norm(gibuu_muons_inrange.genp,axis=1)
    
    #Get weights
    genie_inrange_inds = utils.get_inds_from_sub_inds(set(genie_muons_inrange.index.values),set(genie_nu.index.values),3)
    weights_genie = genie_nu.loc[genie_inrange_inds,'genweight'].values
    
    gibuu_inrange_inds = utils.get_inds_from_sub_inds(set(gibuu_muons_inrange.index.values),set(gibuu_nu.index.values),3)
    weights_gibuu = gibuu_nu.loc[gibuu_inrange_inds,'genweight'].values
    
    #Get number of events from weights
    genie_inrange_count = np.sum(weights_genie)
    gibuu_inrange_count = np.sum(weights_gibuu)
    
    ax.hist([ps_genie,ps_gibuu],
            bins=np.arange(0,4,0.1),
            weights=[weights_genie,weights_gibuu],
            histtype='step',
            lw=2,
            label=[f'GENIE ({round(genie_inrange_count):,})',
                  f'GiBUU ({round(gibuu_inrange_count):,})',],
            density=True if j == 1 else False)
    
    plotters.set_style(ax)
    plotters.add_label(ax,f'{SIM_LABEL}\n{costheta_bins[i]:.2f} < $\cos\\theta_\mu$ < {costheta_bins[i+1]:.2f}',fontsize=12,alpha=0.9,where='centerright')
    ax.legend()
  axs[2,1].set_xlabel(r'$p_\mu$ [GeV]',fontsize=20)
  axs[0,1].set_title(rf'{round(genie_muon_count):,} {GENIE_MODEL} muons & {round(gibuu_muon_count):,} {GIBUU_MODEL} muons',fontsize=25)
  axs[1,0].set_ylabel('Normalized events' if j == 1 else 'Events',fontsize=20)
  if save_plots:
    plotters.save_plot(f'momentum_mu_theta{dens}',fig=fig,folder_name=PLOTS_DIR)





for j,dens in enumerate(['','_dens']):
  fig,axs = plt.subplots(nrows=3,ncols=3,figsize=(15,12),sharex=True)
  for i,ax in enumerate(axs.flatten()):
    #Get masks for momenta ranges
    genie_mask = (np.linalg.norm(genie_muons.genp,axis=1) > p_bins[i]) & (np.linalg.norm(genie_muons.genp,axis=1) < p_bins[i+1])
    gibuu_mask = (np.linalg.norm(gibuu_muons.genp,axis=1) > p_bins[i]) & (np.linalg.norm(gibuu_muons.genp,axis=1) < p_bins[i+1])
    
    #Make dataframes
    genie_muons_inrange = genie_muons[genie_mask]
    gibuu_muons_inrange = gibuu_muons[gibuu_mask]
    
    #Get angles
    costhetas_genie = np.cos(genie_muons_inrange.theta)*np.sign(genie_muons_inrange.theta)
    costhetas_gibuu = np.cos(gibuu_muons_inrange.theta)*np.sign(gibuu_muons_inrange.theta)
    
    #Get weights
    genie_inrange_inds = utils.get_inds_from_sub_inds(set(genie_muons_inrange.index.values),set(genie_nu.index.values),3)
    weights_genie = genie_nu.loc[genie_inrange_inds,'genweight'].values
    
    gibuu_inrange_inds = utils.get_inds_from_sub_inds(set(gibuu_muons_inrange.index.values),set(gibuu_nu.index.values),3)
    weights_gibuu = gibuu_nu.loc[gibuu_inrange_inds,'genweight'].values
    
    #Get number of events from weights
    genie_inrange_count = np.sum(weights_genie)
    gibuu_inrange_count = np.sum(weights_gibuu)
    
    ax.hist([costhetas_genie,costhetas_gibuu],
            bins=np.arange(-1,1,0.1),
            weights=[weights_genie,weights_gibuu],
            histtype='step',
            lw=2,
            label=[f'GENIE ({round(genie_inrange_count):,})',
                  f'GiBUU ({round(gibuu_inrange_count):,})',],
            density=True if j == 1 else False)
    
    plotters.set_style(ax)
    plotters.add_label(ax,f'{SIM_LABEL}\n{p_bins[i]:.2f} < $p_\mu$ < {p_bins[i+1]:.2f}',fontsize=12,alpha=0.9,where='centerleft')
    ax.legend()
  axs[2,1].set_xlabel(r'$\cos \theta_\mu$ [GeV]',fontsize=20)
  axs[0,1].set_title(rf'{round(genie_muon_count):,} {GENIE_MODEL} muons & {round(gibuu_muon_count):,} {GIBUU_MODEL} muons',fontsize=25)
  axs[1,0].set_ylabel('Normalized' if j == 1 else 'Events',fontsize=20)
  if save_plots:
    plotters.save_plot(f'theta_mu_momentum{dens}',fig=fig,folder_name=PLOTS_DIR)

#Single cos range statistics
theta_lower = 0.99
theta_upper = 1
for j,dens in enumerate(['','_dens']):
  fig,ax = plt.subplots(figsize=(8,6))
  #Get muons within theta bins
  genie_muons_inrange = genie_muons[(genie_muons.theta < theta_upper) & (genie_muons.theta > theta_lower)]
  gibuu_muons_inrange = gibuu_muons[(gibuu_muons.theta < theta_upper) & (gibuu_muons.theta > theta_lower)]
  
  #Get momenta
  ps_genie = np.linalg.norm(genie_muons_inrange.genp,axis=1)
  ps_gibuu = np.linalg.norm(gibuu_muons_inrange.genp,axis=1)
  
  #Get weights
  genie_inrange_inds = utils.get_inds_from_sub_inds(set(genie_muons_inrange.index.values),set(genie_nu.index.values),3)
  weights_genie = genie_nu.loc[genie_inrange_inds,'genweight'].values
  
  gibuu_inrange_inds = utils.get_inds_from_sub_inds(set(gibuu_muons_inrange.index.values),set(gibuu_nu.index.values),3)
  weights_gibuu = gibuu_nu.loc[gibuu_inrange_inds,'genweight'].values
  
  #Get number of events from weights
  genie_inrange_count = np.sum(weights_genie)
  gibuu_inrange_count = np.sum(weights_gibuu)
  
  ax.hist([ps_genie,ps_gibuu],
          bins=np.arange(0,3,0.1),
          weights=[weights_genie,weights_gibuu],
          histtype='step',
          lw=3,
          label=[f'GENIE ({round(genie_inrange_count):,})',
                f'GiBUU ({round(gibuu_inrange_count):,})',],
          density=True if j == 1 else False)
  
  ax.legend()
  ax.set_xlabel(r'$p_\mu$ [GeV]',fontsize=20)
  ax.set_ylabel('Normalized events' if j == 1 else 'Events',fontsize=20)
  plotters.set_style(ax)
  plotters.add_label(ax,f'{SIM_LABEL}\n{theta_lower:.2f} < $\cos\\theta_\mu$ < {theta_upper:.2f}',
                     fontsize=20,alpha=0.9,where='centerright',color='darkred')
  if save_plots:
    plotters.save_plot(f'momentum_mu_single_theta_bin_lowerer{dens}',fig=fig,folder_name=PLOTS_DIR)







