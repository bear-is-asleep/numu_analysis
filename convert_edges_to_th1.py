import ROOT
import numpy as np

prism_dir = 'data/prism_fluxes_csv/'

prism_files = [
  'flux_14_oaa_nue_0.0_0.2.csv'
  ,'flux_14_oaa_nue_0.2_0.4.csv'
  ,'flux_14_oaa_nue_0.4_0.6.csv'
  ,'flux_14_oaa_nue_0.6_0.8.csv'
  ,'flux_14_oaa_nue_0.8_1.0.csv'
  ,'flux_14_oaa_nue_1.0_1.2.csv'
  ,'flux_14_oaa_nue_1.2_1.4.csv'
  ,'flux_14_oaa_nue_1.4_1.6.csv'
                ]
prism_files_dir = [prism_dir+f for f in prism_files]

edges = np.arange(0,10.05,0.05)

root_file = ROOT.TFile(f'{prism_dir}prism_fluxes.root','RECREATE')

# Convert each CSV file to a TH1 and save to the ROOT file
for i,csv_f in enumerate(prism_files_dir):
  vals = np.loadtxt(csv_f,delimiter=',')
  h = ROOT.TH1D(f'prism_flux_{i}','',len(edges)-1,edges)
  for j in range(len(vals)):
    h.SetBinContent(j+1,vals[j])
  h.Write()
