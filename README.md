# DeePoreRevised

This repository contains the code for Hope Brooke's final year project, please see the report for context.

This project is built off DeePore, a deep learning workflow for the rapid characterisation of porous materials (https://github.com/ArashRabbani/DeePore). A copy of the original DeePore library has been created, named _DeePoreRevised.py_, which has been extended. All functionality in the file is clearly labelled as new, revised or original. Any new or revised functionality has justifications for it, and revised functionality also includes a brief outline of what has been edited. The original library is included to facilitate comparisons (_DeePore.py_).  

This workflow is designed to be used by others, so while the majority of the functionality is defined in _DeePoreRevised.py_, the process of executing this has been completed separately. These processes are in relevant files:
- **_DatasetCreation.ipynb_** contains the code used to create and validate a new compact dataset from the original 3D material dataset.
- **_ModelTraining.ipynb_** contains the code used to train and test all models defined in the report.
- **_ModelEval.ipynb_** contains the code used to investigate the results and plot graphs to evaluate the models.



Please refer to the comments in these files for more details on how they can be run.

As stated above, the functionality is provided as a library to facilitate the use by others in future. Because of this, a **_Demo.ipynb_** file has been created containing three demos which show how to load a pretrained model to predict properties for a material, and how to retrain and test models of different types, input data slices, and properties.

As Google Colab is used, both the notebook files (_.ipynb_) and the python files (_.py_) are included for all executable code.


## Directory Structure  

### Models
There are 7 models provided:
1. Original DeePore3 (DP3) on 3 slice data on all properties
2. DeePoreRevised1 (DPR1) on 3 slice data on all properties
3. DeePoreRevised2 (DPR2) on 3 slice data on all properties
4. DeePoreRevised1 (DPR1) 6 slice data on all properties
5. DeePoreRevised2 (DPR2) 6 slice data on all properties
6. DeePoreRevised2 (DPR2) 6 slice data on Group 1 properties
7. DeePoreRevised2 (DPR2) 6 slice data on Group 2 properties

Models and related files follow the format of Model**x**_S**y**_P**z**.h5, or Tested_Data_Modelx_Sy_Pz.mat, where...
- **x** is the model number: 3 is DP3; 11 is DPR1; 12 is DPR2. Intermediate numbers are not used as there is still code available to for DP models 1-9, these are not present in the repository but can be trained.
- **y** is the slice number: 1 for one slice in each direction (three in total); 2 for two slices in each direction (six in total).
- **z** is the number of property indices: 1515 for all properties, 1007 for the defined Group 1 and 508 for the defined Group 2. These numbers will vary when training models on different subsets of properties.

Eg. Model**12**_S**2**_P**1515**.h5 is DeePoreRevised scenario 2 trained on the six-slice dataset for all 30 properties.

This is all handled by the library for ease of use - all relevant functions take parameters of the model type, slice number, and a property list.

### Subfolders
**Data/** - This folder contains material data samples provided by the original DeePore project.  
**Images/** - This folder contains the graphs generated for the evaluation of all seven models. Including probability distributions for single values, reference/estimate plots for single values and example reference/estimate plots for functions and distributions.  
**Logs/** - This folder contains all Log files generated by model training processes.

