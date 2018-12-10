# combi_tco
This is a project focused on analysis of triple conducting oxides (TCOs). TCOs are combinatorially synthesized via pulsed laser deposition (PLD) and characterized using high-throughput methods. 

data: raw data from Meagan Papac, collected at NREL
* conductivity: DC conductivity data
* eis: impedance data (ASR)
* xrd: XRD data

scripts: python scripts for processing and analysis
  conductivity: notebooks for DC conductivity analysis
  eis: notebooks for EIS analysis
  helpers: supporting modules
    calc_chemfeat.py: chemical featurization for perovskites
    fileload.py: tools for processing and loading data files
    outlier_detect.py: outlier detection algorithm
    pickletools.py: a few simple tools for working with pickles
    plotting.py: convenience functions for plotting ternary data
    predict.py: functions for predicting through Citrination API
    quaternary_plt.py: quaternary plotting module
  pickles: pickles of python objects for efficiency
  stats_test: early tests of some python stats functions/packages
  xrd: notebooks for XRD analysis
  

  
  
