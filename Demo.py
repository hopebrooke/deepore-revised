
# Below is a provided demo on how to use this workflow
# the outline of it has taken inspiration from the original DeePore Demos

# Dataset route names, and importing the deeporerevised library will need to be adapted for your repository structure

#--------------------------------------------------------------------------------------------------------------------------------

# If using google colab:
# load google drive content
# from google.colab import drive
# drive.mount('/content/drive')
# navigate to project folder - replace with route folder
# %cd /content/drive/MyDrive/hb-final-project

import DeePoreRevised as dpr
import os


# dpr.trainmodel doesn't create Log directory initially
# creating Log directory here to prevent errors
log_dir = '/content/drive/MyDrive/hb-final-project/Logs'  # replace with route for Log folder
os.makedirs(log_dir, exist_ok=True)


dataset_3slice = '/content/drive/MyDrive/hb-final-project-data/DeePore_Compact_Data.h5'   # replace with route to three-slice dataset
dataset_6slice = '/content/drive/MyDrive/hb-final-project-data/DeePoreRevised_Compact_Data.h5'  # replace with route to six-slice dataset



# DEMO 1 - load final model and predict properties
def load_predict_demo():
  # load model
  model = dpr.loadmodel()  # default ModelType = 12, properties=all, and n = 2 (slicenum), but can pass parameters instead to specify
  # feed sample data (replace route location)
  A=dpr.feedsampledata(FileName="/content/drive/MyDrive/hb-final-project/Data/Sample_large.mat")    # default n=2, but can pass parameters instead to specify
  # predict
  preds = dpr.predict(model, A, res=4.8)  # default n=2 and properties=all
  # display result
  dpr.prettyresult(preds, 'results.txt')  # default properties = all



# DEMO 2 - retrain and test model 12 on six-slice dataset
def train_demo_allprops():
  # list prep
  List = dpr.prep(dataset_6slice, n=2)  # change to dataset_3slice and n=1 for three slice dataset
  # split data
  TrainList, EvalList, TestList = dpr.splitdata(List)
  # retrain model
  model = dpr.trainmodel(dataset_6slice, TrainList, EvalList, retrain=0, epochs=20, batch_size=100, ModelType=12, n=2)  # change to dataset_3slice and n=1 for three slice dataset
  # test model
  dpr.testmodel(model, dataset_6slice,TestList, ModelType=12, n=2) # change to dataset_3slice and n=1 for three slice dataset



# DEMO 3 - retrain and test model 12 on six-slice dataset with subgroup of properties
def train_demo_groups():
  properties=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]  # example of retraining on odd numbered properties, this is arbritrary, can choose any out of the 30 and define here
  properties = [x-1 for x in properties]  # change indexing
  # list prep
  List = dpr.prep(dataset_6slice, properties=properties)  # use three slice dataset for n=1
  # split data
  TrainList, EvalList, TestList = dpr.splitdata(List)
  # retrain model, there is no trained model for this set of properties so will have to retrain
  model = dpr.trainmodel(dataset_6slice, TrainList, EvalList, retrain=1, epochs=1, batch_size=100, ModelType=12, properties=properties)
  # test model
  dpr.testmodel(model, dataset_6slice,TestList, ModelType=12, properties=properties)



# uncomment any of the below
# Demo 1
#load_predict_demo()
# Demo 2
#train_demo_allprops()
# Demo 3
#train_demo_groups()
