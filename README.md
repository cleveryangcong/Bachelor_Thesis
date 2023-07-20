# Bachelor Thesis
This thesis deals with the post-processing of data-driven weather models. For this purpose, four different models are implemented and evaluated based on the Continuous Ranked Probability Score (CRPS). The Data available, are the FourCastNet ensemble predictions, for a 120x130 grid over Europe, for the years 2018 - 2022.

#### Data
The data is not publicly available, as it was generated as a separate project at the ECON Institute.

#### Code
The implementation of the models can be found in the src directory. The four models used, are the Ensemble Model Output statics (EMOS) global and local versions, the Distributional Regression Network (DRN) and the U-net.

1. The code for the EMOS models and the DRN are taken and modified from https://github.com/slerch/ppnn, which provides code to the paper: Neural Networks for Postprocessing Ensemble Weather Forecasts (Rasp, Lerch 2018)
2. The implementation of the loss functions, was also taken from https://github.com/slerch/ppnn and modified to fit the use case of this thesis.
3. The code for the U-net is taken from 

#### Figures:
Most of the figures are stored under the reports folder. The code for making the visualizations can be found in the notebooks/reports directory. 






