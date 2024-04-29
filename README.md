# 3931
    
### updatedDP.ipynb:  
Creating a new library with updates to the original DeePore functionality.  

### DeePorePractice.ipynb and DeePorePractice2.ipynb:  
Running, testing, understanding DeePore functionality and Demos provided

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



# Data
- from deepore original

# Images
- Contains first entry visualisation
- Contains probability dist for single value for all 7 models
