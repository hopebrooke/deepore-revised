# DeePoreRevised

There are 7 models provided:

- Original DeePore3 on 3 slice data On all properties
- DeePoreRevised1 on 3 slice data On all properties
- DeePoreRevised2 on 3 slice data On all properties
- DeePoreRevised1 on 6 slice data On all properties
- DeePoreRevised2 on 6 slice data On all properties
                            - again on group 1
                            - again on group 2

Models and related files follow the format of Modelx_Sy_Pz.h5, or Tested_Data_Modelx_Sy_Pz.mat
- where x is model number: 3 for DP3, 11 for DPR1 or 12 for DPR2 (intermediate numbers are not used as there is still code available to for DP models 1-9, these are not present in repository but can be trained
- where y is slice number: 1 for one slice in each direction, three in total; 2 for two slices in each direction, six in total
- where z is number of property indices: 1515 for all properties, 1007 for group 1 and 508 for group 2
Eg. Model12_S2_P1515.h5 is the model trained using deepore revised scenario 2, on the six slice dataset, for all 30 properties
This, however, is all handled by the functions, for training, testing, predicting, loading etc. all functions take parameters ModelType=3 or 11 or 12, n=1 or 2, properties = [list of indices of 30 og properties]
More detail provided in the demo 

### Data
- from deepore original

### Images
- Contains first entry visualisation
- Contains probability dist for single value for all 7 models
- Contains reference/estimate plots for single value for all 7 models
- Contains example for functions/distributions for all 7 models

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
