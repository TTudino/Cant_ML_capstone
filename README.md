# Capstone on the machine learning (ML) use for the prediction of the anthropogenic carbon in the ocean

## Author ##

Tobia Tudino (TTudino)

## Purpose ##

This project provides an initial exploration of the machine learning potentialities to predict the oceanic anthropogenic carbon.
This aims to: 
  - find an alternative approach to the currently used methods,
  - reduce the uncertainty in the anthropogenic carbon estimates (currently at approximately 20%).

## Related reading ##

[1] Friis, K., Körtzinger, A., Pätsch, J., and Wallace, D.W.R.: On the temporal increase of anthropogenic CO2 in the subpolar North Atlantic, Deep Sea Res. I, 52, 681-698, 2005.

[2] Gruber, N., Sarmiento, J.L., and Stocker, T.F.: An improved method for detecting anthropogenic CO2 in the oceans, Glob. Biogeochem. Cycles, 10, 809-837, 1996.

[3] Khatiwala, S., Primeau, F., and Hall, T.: Reconstruction of the history of anthropogenic CO2 concentrations in the ocean, Nature, 462, 346-349, 2009.

[4] S.K. Lauvset et al. A new global interior ocean mapped climatology: the 1◦ x 1◦ GLODAP version 2. Earth Syst. Sci. Data, 8, 2016. doi: 10.5194/essd-8-325-2016.

[5] Redfield, A.C.: On the proportions of organic derivations in seawater and their relation to the composition of Plankton. In J. Johnstone memorial, Liverpool University press, 176-192, 1934.

[6] Sabine, C.L., Feely, R.A., Gruber, N., Key, R.M., Lee, K., Bullister, J.L., Wanninkhof, R., Wong, C.S., Wallace, D.W.R., Tillbrock, B., Millero, F.J., Peng, T.-H., Kozyr, A., Ono, T., and Rios, A.F.: The Oceanic Sink for Anthropogenic CO2, Science, 305, 367-371, 2004.

[7] Waugh, D.W., Haine, T.W.N., and Hall, T.M.: Transport times and anthropogenic carbon in the subpolar North Atlantic Ocean, Deep Sea Res. I, 51, 1475-1491, 2004.

[8] https://www.kdnuggets.com/2018/04/right-metric-evaluating-machine-learning-models-1.html

[9] http://www.mvstat.net/tduong/research/seminars/seminar-2001-05/

## Dependencies ##

To limit the size of this repository, the GLODAPv2 climatology dataset is not provided. 
To obtain those data, please refer to https://www.glodap.info/, read the article [4] as above, or contact me directly.

## Use ##

# Initial analysis #

The initial analysis is summarised in the `capstone_project.pdf` file. 

Please read it through and feel free to ask/suggest any change.

# Code use #

The code is provided as jupyter notebook (`ML_Cant_estimate.ipynb`) and python code (`ML_Cant_estimate.py`). Adjust the existing paths to data where necessary.

The code is set to run on a small subset of data randomly extracted from the provided dataset. Remove this limitations if your machine is powerful enough to deal with a bigger dataset. The results improve.
