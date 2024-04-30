
# Below is the code used to train all models on Google Colab
# The functionality for it is defined in the DeePoreRevised file

# Runtime to train models for 100 epochs on three-slice data is ~6.75 hours
# Runtime to train models for 145 epochs on six-slice data is ~17 hours

# While the models are created already and available, this code can be run to retrain or reload and retrain to verify the correctness
# Dataset route names, and importing the deeporerevised library will need to be adapted for your repository structure
# If you set retrain=1, the models will retrain from the beginning, so be aware that any previous models may be written over

# Please refer to the readme for more clarity on how models and relevant files are named

#-----------------------------------------------------------------------------------------------------------------------------------

# If using google colab:
# load google drive content
# from google.colab import drive
# drive.mount('/content/drive')
# navigate to project folder - replace with relevant project folder
# %cd /content/drive/MyDrive/hb-final-project

import DeePoreRevised as dpr
import os

# dpr.trainmodel doesn't create Log directory initially
# creating Log directory here to prevent errors
log_dir = '/content/drive/MyDrive/hb-final-project/Logs'  # replace with route for Log folder
os.makedirs(log_dir, exist_ok=True)

dataset_3slice = '/content/drive/MyDrive/hb-final-project-data/DeePore_Compact_Data.h5'   # replace with route to three-slice dataset
dataset_6slice = '/content/drive/MyDrive/hb-final-project-data/DeePoreRevised_Compact_Data.h5'  # replace with route to six-slice dataset




# TRAINING ORIGINAL DEEPORE3 (DP3) - MODEL 3, ON 3 SLICE, ALL PROPERTIES
# prepare dataset
List=dpr.prep(dataset_3slice, 1)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_3slice,TrainList,EvalList,retrain=0,reload=0,epochs=100,batch_size=100,ModelType=3, n=1)
# test model
dpr.testmodel(model,dataset_3slice,TestList, ModelType=3, n=1)




# TRAINING DEEPOREREVISED1 (DPR1) - MODEL 11, ON 3 SLICE, ALL PROPERTIES
# prepare dataset
List=dpr.prep(dataset_3slice, 1)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_3slice,TrainList,EvalList,retrain=0,reload=0,epochs=100,batch_size=100,ModelType=11, n=1)
# test model
dpr.testmodel(model,dataset_3slice,TestList, ModelType=11, n=1)




# TRAINING DEEPOREREVISED1 (DPR1) - MODEL 11, ON 6 SLICE, ALL PROPERTIES
# prepare dataset
List=dpr.prep(dataset_6slice, 2)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_6slice,TrainList,EvalList,retrain=0,reload=0,epochs=145,batch_size=100,ModelType=11, n=2)
# test model
dpr.testmodel(model,dataset_6slice,TestList, ModelType=11, n=2)





# TRAINING DEEPOREREVISED2 (DPR2) - MODEL 12, ON 3 SLICE, ALL PROPERTIES
# prepare dataset
List=dpr.prep(dataset_3slice, 1)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_3slice,TrainList,EvalList,retrain=0,reload=0,epochs=100,batch_size=100,ModelType=12, n=1)
# test model
dpr.testmodel(model,dataset_3slice,TestList, ModelType=12, n=1)





# TRAINING DEEPOREREVISED2 (DPR2) - MODEL 12, ON 6 SLICE, ALL PROPERTIES
# prepare dataset
List=dpr.prep(dataset_6slice, 2)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_6slice,TrainList,EvalList,retrain=0,reload=0,epochs=145,batch_size=100,ModelType=12, n=2)
# test model
dpr.testmodel(model,dataset_6slice,TestList, ModelType=12, n=2)





# TRAINING DEEPOREREVISED2 (DPR2) - MODEL 12, ON 6 SLICE, GROUP 1 PROPERTIES
# define group 1
group_1 = [1,4,7,8,9,10,12,16,17,18,20,21,22,23,24,25,26]
# iterate through subtracting 1 from each
group_1 = [x-1 for x in group_1]
# prepare dataset
List=dpr.prep(dataset_6slice, 2)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_6slice,TrainList,EvalList,retrain=0,reload=0,epochs=145,batch_size=100,ModelType=12, n=2, properties=group_1)
# test model
dpr.testmodel(model,dataset_6slice,TestList, ModelType=12, n=2, properties=group_1)





# TRAINING DEEPOREREVISED2 (DPR2) - MODEL 12, ON 6 SLICE, GROUP 2 PROPERTIES
# define group 2
group_2 = [2,3,5,6,11,13,14,15,19,27,28,29,30]
# iterate through subtracting 1 from each
group_2 = [x-1 for x in group_2]
# prepare dataset
List=dpr.prep(dataset_6slice, 2)
# split data 80-10-10
TrainList, EvalList, TestList = dpr.splitdata(List)
# train model - set retrain and/or reload to 1 to retrain and write over model file
model=dpr.trainmodel(dataset_6slice,TrainList,EvalList,retrain=0,reload=0,epochs=145,batch_size=100,ModelType=12, n=2, properties=group_2)
# test model
dpr.testmodel(model,dataset_6slice,TestList, ModelType=12, n=2, properties=group_2)
