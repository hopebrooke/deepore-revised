# DeePoreRevised

This repository contains the code for Hope Brooke's final year project, please see the report for context.

## Models
There are 7 models provided:
1. Original DeePore3 on 3 slice data on all properties
2. DeePoreRevised1 on 3 slice data on all properties
3. DeePoreRevised2 on 3 slice data on all properties
4. DeePoreRevised1 on 6 slice data on all properties
5. DeePoreRevised2 on 6 slice data on all properties
6. DeePoreRevised2 on 6 slice data on Group 1 properties
7. DeePoreRevised2 on 6 slice data on Group 2 properties

Models and related files follow the format of Modelx_Sy_Pz.h5, or Tested_Data_Modelx_Sy_Pz.mat, where...
- x is model number: 3 for DP3, 11 for DPR1 or 12 for DPR2 (intermediate numbers are not used as there is still code available to for DP models 1-9, these are not present in repository but can be trained
- y is slice number: 1 for one slice in each direction, three in total; 2 for two slices in each direction, six in total
- z is number of property indices: 1515 for all properties, 1007 for group 1 and 508 for group 2

Eg. Model12_S2_P1515.h5 is the model trained using deepore revised scenario 2, on the six slice dataset, for all 30 properties

This is all handled by the functions for ease of use. All relevant functions take parameters: ModelType=3 or 11 or 12 (default); n=1 or 2 (default); properties = [list of indices of 30 properties] eg. [1,5,11,15,19], default is all 30.

## Subfolders
**Data/ - ** This folder contains material data samples provided by the original DeePore project

#### Images
This folder contains the graphs generated for the evaluation of all seven models. Including probability distributions for single values, reference/estimate plots for single values and example reference/estimate plots for functions and distributions.

### Logs
- creates training model logs, as models were trained in steps using reload, these logs are split over ultiple files for each model

### Other
- DeePore.py is contained purely as a reference
- All new files have both .py and .ipynb versions of the same code for ease of use
- DeePoreRevised.py is new library, all code is clearly marked as new, revised, or original deepore
- DatasetCreation is the code used to create and validate new functionality
- ModelTraining is the code used to train and test all 7 models
- ModelEval provides the code used to plot all evaluation functions
- Demo.py has 3 demos which show how to predict properties for a material, and how to retrain and test models of different types, input data slices and proeprties.
