
# REPEAT OF DEEPORE.PY WITH NEW AND REVISED FUNCTIONS
# ALL FUNCTIONS WILL BE COMMENTED TO OUTLINE WHICH ARE ORIGINAL, REVISED, OR NEW
# THE ORIGINAL DEEPORE FILE/LIBRARY WILL BE AVAILABLE FOR COMPARISON

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Add
from tensorflow.keras.models import Model
import os, sys
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import scipy.io as sio
from scipy.ndimage import distance_transform_edt as distance
import tensorflow.keras.backend as kb

# REVISED DEEPORE
# Reason -> problem with float/int conversion, 6 channels cannot be visualised as an RGB image
# Revision -> added int casting, and doesn't generate 6 channel data as RGB
def show_feature_maps(A, n=2):
    if n != 1:
      print("Cannot visualise feature maps with 6 channels as an RGB image.")
      return
    N=int(np.ceil(np.sqrt(A.shape[0])))
    f=plt.figure(figsize=(N*10,N*10))
    for I in range(A.shape[0]):
        plt.subplot(N,N,I+1)
        plt.imshow(normal(np.squeeze(A[I,:,:,:])))
        plt.axis('off')
    plt.show()

    f.savefig('images/initial_feature_maps.png')

# NEW FUNCTION
# Reason -> directs to correct slive vol func based on how many slices requested
def slicevol(A, n):
  if n == 1:
    return slicevol_1(A)
  elif n == 2:
    return slicevol_2(A)
  else:
    print("Error: Not a valid slice number.")

# ORIGINAL DEEPORE
def slicevol_1(A):
    A=np.squeeze(A)
    B=np.zeros((1,A.shape[0],A.shape[1],3))
    B[0,:,:,0]=A[int(A.shape[0]/2),:,:]
    B[0,:,:,1]=A[:,int(A.shape[1]/2),:]
    B[0,:,:,2]=A[:,:,int(A.shape[2]/2)]
    return B

# REVISED DEEPORE
# Reason -> editing original slicevol to take 2 slices in each direction instead of 1
# Revision -> 2 slices taken from each direction at 1/3 and 2/3, rather than at 1/2
def slicevol_2(A):
    A=np.squeeze(A)
    B=np.zeros((1,A.shape[0],A.shape[1],6))
    # slices from front
    B[0,:,:,0]=A[int(A.shape[0]/3),:,:]
    B[0,:,:,1]=A[int(2 * A.shape[0]/3),:,:]
    # slices from left
    B[0,:,:,2]=A[:,int(A.shape[1]/3),:]
    B[0,:,:,3]=A[:,int(2 * A.shape[1]/3),:]
    # slices from top
    B[0,:,:,4]=A[:,:,int(A.shape[2]/3)]
    B[0,:,:,5]=A[:,:,int( 2 * A.shape[2]/3)]
    return B

# NEW FUNCTION
# Reason -> directs to correct show entry function based on how many slices requested
def showentry(A, n):
  if n == 1:
    showentry_1(A)
  elif n == 2:
    showentry_2(A)
  else:
    print("Error: Not a valid slice number.")

# REVISED DEEPORE
# Reason -> np.int deprecated
# Revision -> replaced np.int with int
def showentry_1(A):
    """shows 3 slices of a volume data """
    A=np.squeeze(A)
    plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
    CM=plt.cm.viridis
    ax1=plt.subplot(1,3,1); plt.axis('off'); ax1.set_title('X mid-slice')
    plt.imshow(np.squeeze(A[int(A.shape[0]/2), :,:]), cmap=CM, interpolation='nearest')
    ax2=plt.subplot(1,3,2); plt.axis('off'); ax2.set_title('Y mid-slice')
    plt.imshow(np.squeeze(A[:,int(A.shape[1]/2), :]), cmap=CM, interpolation='nearest')
    ax3=plt.subplot(1,3,3); plt.axis('off'); ax3.set_title('Z mid-slice');
    plt.imshow(np.squeeze(A[:,:,int(A.shape[2]/2)]), cmap=CM, interpolation='nearest')
    plt.savefig('images/First_entry_1.png')

# REVISED DEEPORE
# Reason -> changing slice num means we need to change how the entries are shown
# Revision -> shows 2 slices in each direction, at 1/3 and 2/3 of the shape
def showentry_2(A):
    """shows 6 slices of a volume data """
    A=np.squeeze(A)
    plt.figure(num=None, figsize=(20, 14), dpi=80, facecolor='w', edgecolor='k')
    CM=plt.cm.viridis
    # X SLICES
    ax1=plt.subplot(2,3,1); plt.axis('off'); ax1.set_title('X mid-slice 1')
    plt.imshow(np.squeeze(A[int(A.shape[0]/3), :,:]), cmap=CM, interpolation='nearest')
    ax2=plt.subplot(2,3,4); plt.axis('off'); ax2.set_title('X mid-slice 2')
    plt.imshow(np.squeeze(A[int(2 * A.shape[0]/3), :,:]), cmap=CM, interpolation='nearest')
    # Y SLICES
    ax3=plt.subplot(2,3,2); plt.axis('off'); ax3.set_title('Y mid-slice 1')
    plt.imshow(np.squeeze(A[:,int(A.shape[1]/3), :]), cmap=CM, interpolation='nearest')
    ax4=plt.subplot(2,3,5); plt.axis('off'); ax4.set_title('Y mid-slice 2')
    plt.imshow(np.squeeze(A[:,int(2 * A.shape[1]/3), :]), cmap=CM, interpolation='nearest')
    # Z SLICES
    ax5=plt.subplot(2,3,3); plt.axis('off'); ax5.set_title('Z mid-slice 1');
    plt.imshow(np.squeeze(A[:,:,int(A.shape[2]/3)]), cmap=CM, interpolation='nearest')
    ax6=plt.subplot(2,3,6); plt.axis('off'); ax6.set_title('Z mid-slice 2');
    plt.imshow(np.squeeze(A[:,:,int(2 * A.shape[2]/3)]), cmap=CM, interpolation='nearest')
    plt.savefig('images/First_entry_2.png')

# REVISED DEEPORE
# Reason -> needs to also know how many slices being requested in each dir (1, 2 or 3), adjust saving so that it is in uint8 range and can later be returned to -1->1 range
# Revision -> directs to correct slice vol/ecl_distance function, changes shape of h5 slice written, calulated euclidean distance and normalisation differently to be saved in uint8 format
def create_compact_dataset(Path_complete,Path_compact, n=2):
    S=hdf_shapes(Path_complete,['X'])
    for I in range(S[0][0]):
        X=readh5slice(Path_complete,'X',[I])
        Y=readh5slice(Path_complete,'Y',[I])
        if n == 1:
          X=slicevol_1(X)
        elif n == 2:
          X=slicevol_2(X)
        else:
          print("Functionality for that quantity of slices has not been implemented.")
        X=ecl_distance(X, n)

        B=np.zeros((X.shape[0],128,128,3*n))
        for I in range(X.shape[0]):
            for J in range(X.shape[3]):
                t=distance(np.squeeze(X[I,:,:,J]))  - distance(np.squeeze(1-X[I,:,:,J]))

                t[t<0]= 0

                t = MaxPooling2D((2, 2)) (np.reshape(t,(1,256,256,1)))
                t=np.float64(t)
                t = t * 13.5
                t = np.floor(t) # rescale for right format
                B[I,:,:,J]=np.squeeze(t)

        writeh5slice(B.astype(np.uint8),Path_compact,'X',Shape=[128,128,3*n])
        writeh5slice(Y,Path_compact,'Y',Shape=[1515,1])

# REVISED DEEPORE
# Reason -> need to change names of models saved for different slices, change output shape for different property numbers
# Revision -> model and minmax file name changes depending on slices, number of properties to predict is calculated
def loadmodel(ModelType=12, properties=None, n=2):    # default model throughout is 12 (dpr2), and n=2
    # calculate property num
    property_num=0
    if properties is None:
      property_num = 1515
    else:
      for prop in properties:
        if prop < 15:
          property_num += 1
        else:
          property_num += 100
    Path='Model'+str(ModelType)+'_S'+str(n)+'_P'+str(property_num)+'.h5';
    MIN,MAX=np.load('minmax_s'+str(n)+'_p'+str(property_num)+'.npy')
    slices = 3*n
    INPUT_SHAPE=[1,128,128,slices];
    OUTPUT_SHAPE=[1,property_num,1];
    model=modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType,property_num)
    model.load_weights(Path)
    return model

# REVISED DEEPORE
# Reason -> ecl distance calculations need to be changed for different slice volumes
# Revision -> adjusted shape arrays for 6 instead of 3
def ecl_distance(A, n=2):
    B=np.zeros((A.shape[0],128,128,3*n))
    for I in range(A.shape[0]):
        for J in range(A.shape[3]):
            t=distance(np.squeeze(1-A[I,:,:,J]))-distance(np.squeeze(A[I,:,:,J]))
            # t=normalize(t)
            t=np.float32(t)/64

            t[t>1]=1
            t[t<-1]=-1

            t = MaxPooling2D((2, 2)) (np.reshape(t,(1,256,256,1)))
            t=np.float64(t)
            B[I,:,:,J]=np.squeeze(t)
    return B

# REVISED DEEPORE
# Reason -> seperate minmax files for different slice quantities, need to only record the properties wanted
# Revision -> takes slice num as parameter n, loads minmax_<n>.py, takes properties list as parameter,
#             retrieve only properties specified, calculates min / max appropriately, use num of properties when setting MIN/MAX
def prep(Data, n=2, properties=None):

    num_single_vals = 0
    num_range_vals = 0
    if properties is None:
      properties = list(range(1515))
      num_single_vals = 15
      num_range_vals = 15
    else:
      # map the properties from 1-30 to 1-1515
      mapped_properties = []
      for prop in properties:
        # if under 15 just add to new array
        if 0 <= prop < 15:
          mapped_properties.append(prop)
          num_single_vals += 1
        # if over 15, add mapped 100 values
        elif prop >= 15:
          val = ((prop-15)*100)+15
          num_range_vals += 1
          for i in range (0, 100):
            mapped_properties.append(val+i)
      properties=mapped_properties

    print('Checking the data for outliers. Please wait...')
    List=[]
    with h5py.File(Data,'r') as f:
        length=f['X'].shape[0]
        MIN=np.ones((len(properties),1))*1e7
        MAX=-MIN
        counter=0
        for I in range(length):
            t2=f['Y'][counter, properties]
            y=t2.astype('float32')
            D=int(np.sum(np.isnan(y)))+int(np.sum(np.isinf(y)))
            # check if specific properties are chosen - and whether they are valid
            if 1 in properties:
              D += int(y[properties.index(1)] > 120)
            if 4 in properties:
              D += int(y[properties.index(4)]> 1.8)
            if 0 in properties:
              D += int(y[properties.index(0)]< 1e-4)
            if 2 in properties:
              D += int(y[properties.index(2)]< 1e-5)
            if 14 in properties:
              D += int(y[properties.index(14)]> 0.7)

            if D>0:
                pass
            else:
                List=np.append(List,counter)
                y[0:num_single_vals]=np.log10(y[0:num_single_vals]) # applying log10 to handle range of order of magnitudes
                maxid=np.argwhere(y>MAX)
                minid=np.argwhere(y<MIN)
                MAX[maxid[:,0]]=y[maxid[:,0]]
                MIN[minid[:,0]]=y[minid[:,0]]
            if counter % 100==0:
                print('checking sample: '+str(counter))
            counter=counter+1

        Singles=num_single_vals
        for I in range(num_range_vals):
            MAX[Singles+100*I:Singles+100*(I+1)]=np.max(MAX[Singles+100*I:Singles+100*(I+1)])
            MIN[Singles+100*I:Singles+100*(I+1)]=np.min(MIN[Singles+100*I:Singles+100*(I+1)])
    np.save('minmax_s'+str(n)+'_p'+str(len(properties))+'.npy',[MIN,MAX])
    return List

# REVISED DEEPORE
# Reason -> different slice volumes have different minmax and model names, takes list of properties to predict
# Revision -> takes slice vol as parameter n (defaults to 1 if not), and loads/saves appropriate minmax, logs, models, new parameter of properties
def trainmodel(DataName,TrainList,EvalList,retrain=0,reload=0,epochs=100,batch_size=100,ModelType=12, n=2, properties=None):

    condensed_properties = properties
    if properties is None:
      properties = list(range(1515))
    else:
      # map the properties from 1-30 to 1-1515
      mapped_properties = []
      for prop in properties:
        # if under 15 just add to new array
        if 0 <= prop < 15:
          mapped_properties.append(prop)
        # if over 15, add mapped 100 values
        elif prop >= 15:
          val = ((prop-15)*100)+15
          for i in range (0, 100):
            mapped_properties.append(val+i)
      properties=mapped_properties

    from tensorflow.keras.callbacks import ModelCheckpoint
    MIN,MAX=np.load('minmax_s'+str(n)+'_p'+str(len(properties))+'.npy')
    SaveName='Model'+str(ModelType)+'_S'+str(n)+'_P'+str(len(properties))+'.h5';
    INPUT_SHAPE,OUTPUT_SHAPE =hdf_shapes(DataName,('X','Y'));
    OUTPUT_SHAPE=[1,1]
    # callbacks
    timestr=nowstr()
    LogName='log_'+timestr+'_'+'Model'+str(ModelType) + '_S' + str(n) + '_P' + str(len(properties))
    filepath=SaveName
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_freq=50, save_best_only=True, mode='min')
    with open("Logs/"+LogName+".txt", "wt") as f:
        f.write('# Path to train file: \n')
        f.write(DataName +'\n')
        f.write('# Start time: \n')
        f.write(timestr +'\n')
        nowstr()
        st='# Training loss'
        spa=' ' * (40-len(st))
        st=st+spa+'Validation loss'
        f.write(st+'\n')

    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.val_loss_=None
            self.start_time=now()
        def on_batch_end(self, batch, logs=None):
            if self.val_loss_==None:
                self.val_loss_=logs['mse']
            with open("Logs/"+LogName+".txt", "a") as f:
                st=str(logs['mse'])
                spa=' ' * (40-len(st))
                st=st+spa+str(self.val_loss_)
                f.write(st+'\n')
        def on_test_batch_end(self,batch, logs=None):
            self.val_loss_=logs['mse']
    callbacks_list = [checkpoint,MyCallback()]
    callbacks_list = [checkpoint]
    callbacks_list = []
    callbacks_list = [MyCallback()]
    # end of callbacks
    model=modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType, len(properties))


    if retrain:
        if reload:
            model.load_weights(SaveName)

        model.fit(gener(batch_size,DataName,TrainList,MIN,MAX, condensed_properties), epochs=epochs,steps_per_epoch=int(len(TrainList)/batch_size),
                  validation_data=gener(batch_size,DataName,EvalList,MIN,MAX, condensed_properties),validation_steps=int(len(EvalList)/batch_size),callbacks=callbacks_list)
        model.save_weights(SaveName);
    else:
        model.load_weights(SaveName)
    return model

# REVISED DEEPORE
# Reason -> changing load/save names depending on model slices, need to define how many single value properties are being predicted, change which variable names are shown based on which properties are predicted
# Revision -> Takes slice vol as parameter n, changes minmax load name and tested model save name, takes properties predicted as parameter, uses num of single value instead of 15, and uses list to determine which variable names are used
#            changed graph so x and y axes are same.
def testmodel(model,DataName,TestList,ModelType=12, n=2, properties=None):
    condensed_properties = properties
    num_single_vals = 0
    num_range_vals = 0
    if properties is None:
      properties = list(range(1515))
      num_single_vals = 15
      num_range_vals = 15
      condensed_properties=list(range(30))
    else:
      # map the properties from 1-30 to 1-1515
      mapped_properties = []
      for prop in properties:
        # if under 15 just add to new array
        if 0 <= prop < 15:
          mapped_properties.append(prop)
          num_single_vals += 1
        # if over 15, add mapped 100 values
        elif prop >= 15:
          val = ((prop-15)*100)+15
          num_range_vals += 1
          for i in range (0, 100):
            mapped_properties.append(val+i)
      properties=mapped_properties

    MIN,MAX=np.load('minmax_s'+str(n)+'_p'+str(len(properties))+'.npy')
    G=gener(len(TestList),DataName,TestList,MIN,MAX,condensed_properties)
    L=next(G)
    x=L[0]
    y=L[1]
    y2=model.predict(L[0])
    print('\n# Evaluate on '+ str(TestList.shape[0]) + ' test data')
    model.evaluate(x,y,batch_size=50)
    #  Denormalize the predictions
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y2=np.multiply(y2,(MAX-MIN))+MIN
    y[:,0:num_single_vals]=10**y[:,0:num_single_vals]
    y2[:,0:num_single_vals]=10**y2[:,0:num_single_vals]

    # save test results as mat file for postprocessing with matlab
    import scipy.io as sio
    sio.savemat('Tested_Data_Model'+str(ModelType)+'_S'+str(n)+'_P'+str(len(properties))+'.mat',{'y':y,'y2':y2})
    # #  Show prediction of single-value features
    fig=plt.figure(figsize=(30,40))
    plt.rcParams.update({'font.size': 30})
    with open('VarNames.txt') as f:
        VarNames = list(f)
    for I in range(num_single_vals):
        ax = fig.add_subplot(5,3,I+1)
        X=y[:,I]
        Y=y2[:,I]
        plt.scatter(X,Y)
        plt.ylabel('Predicted')
        plt.xlabel('Ground truth')
        plt.tick_params(direction="in")
        plt.text(.5,.9,VarNames[properties[I]],horizontalalignment='center',transform=ax.transAxes) # make sure to get relevant var names if subset of properties
        min_val = min(np.min(X), np.min(Y))
        max_val = max(np.max(X), np.max(Y))
        plt.xlim(min_val,max_val)
        plt.ylim(min_val,max_val)
        if I==0:
            ax.set_yscale('log')
            ax.set_xscale('log')
    plt.savefig('images/Single-value_Features'+str(ModelType)+'_S'+str(n)+'_P'+str(len(properties))+'.png')

# REVISED DEEPORE
# Reason -> slice volume affects how ecl_distance is calulated
# Revision -> takes slice vol as parameter n and passes parameter to ecl_distance
def feedsampledata(A=None,FileName=None, n=2):

    if n < 1 or n > 2:
      print("Functionality for that slice volume has not been implemented. Please use n=1 for one slice along each axis, or n=2 for two slices along each axis.")
      return

    if FileName!=None:
        extention=os.path.splitext(FileName)[1]
    else:
        extention=''
    if extention=='.mat':
        # A=mat2np(FileName)
        import scipy.io as sio
        A=sio.loadmat(FileName)['A']
    if extention=='.npy':
        A=np.load(FileName)
    if extention=='.npz':
        A=np.load(FileName)['A']
    if extention=='.npy' or extention=='.mat' or extention=='.npz' or FileName==None:
        if len(A.shape)==3:

            A=np.int8(A!=0)
            LO,HI=makeblocks(A.shape,w=256,ov=.1)
            N=len(HI[0])*len(HI[1])*len(HI[2]) # number of subsamples

            AA=np.zeros((N,256,256,3*n))
            a=0
            for I in range(len(LO[0])):
                for J in range(len(LO[1])):
                    for K in range(len(LO[2])):
                      temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J],LO[2][K]:HI[2][K]]
                      if n == 1:
                        temp1=np.squeeze(temp[int(temp.shape[0]/2),:,:]);
                        temp2=np.squeeze(temp[:,int(temp.shape[1]/2),:]);
                        temp3=np.squeeze(temp[:,:,int(temp.shape[2]/2)]);
                        AA[a,...]=np.stack((temp1,temp2,temp3),axis=2)
                        a=a+1
                      elif n == 2:
                        temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J],LO[2][K]:HI[2][K]]
                        temp1=np.squeeze(temp[int(temp.shape[0]/3),:,:]);
                        temp2=np.squeeze(temp[2*int(temp.shape[0]/3),:,:]);
                        temp3=np.squeeze(temp[:,int(temp.shape[1]/3),:]);
                        temp4=np.squeeze(temp[:,2*int(temp.shape[1]/3),:]);
                        temp5=np.squeeze(temp[:,:,int(temp.shape[2]/3)]);
                        temp6=np.squeeze(temp[:,:,2*int(temp.shape[2]/3)]);
                        AA[a,...]=np.stack((temp1,temp2,temp3,temp4,temp5,temp6),axis=2)
                        a=a+1

    if extention=='.png' or extention=='.jpg' or extention=='.bmp':
        A=plt.imread(FileName)
        if len(A.shape)!=2:
            print('Converting image to grayscale...')
            A=np.mean(A,axis=2)
            print('Converting image to binary...')
            import cv2
            ret,A = cv2.threshold(A,127,255,cv2.THRESH_BINARY)
        A=np.int8(A!=0)
        LO,HI=makeblocks(A.shape,w=256,ov=.1)
        N=len(HI[0])*len(HI[1]) # number of subsamples
        AA=np.zeros((N,256,256,3*n))
        a=0
        for I in range(len(LO[0])):
            for J in range(len(LO[1])):
                temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J]]
                if n == 1:
                  AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)
                else:
                  AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1),temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)  # add double of each direction for 6 slices of 3d images
                a=a+1
    if FileName==None:
        if len(A.shape)==2:
            A=np.int8(A!=0)
            LO,HI=makeblocks(A.shape,w=256,ov=.1)
            N=len(HI[0])*len(HI[1]) # number of subsamples
            AA=np.zeros((N,256,256,3*n))
            a=0
            for I in range(len(LO[0])):
                for J in range(len(LO[1])):
                    temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J]]
                    if n == 1:
                      AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)
                    else:
                      AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1),temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)  # add double of each direction for 6 slices of 3d images
                    a=a+1

    B=ecl_distance(AA, n)
    return B

# REVISED DEEPORE
# Reason -> Minmax depends on slice volume, need array of values that are in properties predicted
# Revision -> takes slice vol as parameter n and adjust file loading accordingly, maps properties predicted to y responses from model
def predict(model,A,res=5, n=2, properties=None):

    # set to 30 (all 1515) if not specified
    if properties is None:
      properties = list(range(30))
    # calculate the number of single values
    num_single_vals = sum(1 for prop in properties if prop < 15)
    num_range_vals = sum(1 for prop in properties if prop >= 15)

    MIN,MAX=np.load('minmax_s'+str(n)+'_p'+str(num_single_vals+100*num_range_vals)+'.npy')
    y=model.predict(A)
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y=np.mean(y,axis=0)

    values = y[0:num_single_vals]

    # single values
    for i in range (0, num_single_vals):
      prop = properties[i]
      # normalise single values
      if prop < 15:
        values[i]=10**y[i]
        # loop to check which prooperties are being predcted
        if prop == 0:
          values[i] = values[i] * res * res
        elif prop == 3:
          values[i] = values[i] / res / res / res
        elif prop == 10:
          values[i] = values[i] / res
        elif prop in [6,7,8,13]:
          values[i] = values[i] * res
    output = values
    d=100
    # loop through each range
    for i in range (0, num_range_vals):
      # get that range based on how many single and range values calculated:
      func=y[(i*d)+num_single_vals:(i+1)*d + num_single_vals]
      # below means results cannot be compared to original deepore values as they are not scaled
      if properties[i+num_single_vals] in [19,20,21,22,23,24,29]:
        func = func * res
      if properties[i+num_single_vals] in [18]:
        func = func / res
      if properties[i+num_single_vals] in [25]:
        func = func * res * res
      output = np.append(output, func)

    return output

# REVISED DEEPORE
# Reason -> different groupings of properties affects how this function works
# Revision -> takes properties list as parameter
def gener(batch_size,Data,List,MIN,MAX, properties=None):

    num_single_vals = 0
    num_range_vals = 0
    if properties is None:
      properties = list(range(1515))
      num_single_vals = 15
      num_range_vals = 15
    else:
      # map the properties from 1-30 to 1-1515
      mapped_properties = []
      for prop in properties:
        # if under 15 just add to new array
        if 0 <= prop < 15:
          mapped_properties.append(prop)
          num_single_vals += 1
        # if over 15, add mapped 100 values
        elif prop >= 15:
          val = ((prop-15)*100)+15
          num_range_vals += 1
          for i in range (0, 100):
            mapped_properties.append(val+i)
      properties=mapped_properties

    with h5py.File(Data,'r') as f:
        length=len(List)
        samples_per_epoch = length
        number_of_batches = int(samples_per_epoch/batch_size)
        counter=0
        while 1:
            t1=f['X'][np.int32(np.sort(List[batch_size*counter:batch_size*(counter+1)])),...]
            t2=f['Y'][np.int32(np.sort(List[batch_size*counter:batch_size*(counter+1)])),...]
            t2 = t2[:, properties, ...]# only get properties defined
            X_batch=t1.astype('float32')/128
            y_batch=t2.astype('float32')
            y_batch=np.reshape(y_batch,(y_batch.shape[0],y_batch.shape[1]))
            y_batch[:,0:num_single_vals]=np.log10(y_batch[:,0:num_single_vals])
            Min=np.tile(np.transpose(MIN),(batch_size,1))
            Max=np.tile(np.transpose(MAX),(batch_size,1))
            y_batch=(y_batch-Min)/(Max-Min)
            counter += 1
            ids=shuf(np.arange(np.shape(y_batch)[0]))
            X_batch=X_batch[ids,...]
            y_batch=y_batch[ids,...]
            yield X_batch,y_batch
            if counter >= number_of_batches:
                counter = 0

# REVISED DEEPORE
# Reason -> need to change which prediction values are displayed based on what properties have been predicted
# Revision -> Takes property list as parameter, used to know how many to print, and which variable name to associate with what prediction, scaling needs to be adjusted
def prettyresult(vals,FileName,units='um',verbose=1, properties=None):

    # set to 30 (all 1515) if not specified
    if properties is None:
      properties = list(range(30))
    # calculate the number of single values
    num_single_vals = sum(1 for prop in properties if prop < 15)
    num_range_vals = sum(1 for prop in properties if prop >= 15)

    vals=np.squeeze(vals)
    with open('VarNames.txt') as f:
        VarNames = list(f)
    b=np.round(vals[0:num_single_vals],7)
    f = open(FileName, 'w')
    f.write('DeePore output results including 15 single-value' +'\n'+ 'paramters, 4 functions and 11 distributions'+'\n')
    f.write('_' * 50+'\n')
    f.write('        ### Single-value parameters ###'+'\n')
    f.write('_' * 50+'\n')
    f.write('\n')
    t='Properties'
    spa=' ' * (40-len(t))
    f.write(t+spa+'Value'+'\n')
    f.write('-' * 50+'\n')
    for i in range(len(b)):
        t=VarNames[properties[i]].strip()
        if units=='um':
            t=t.replace('px','um')
        spa=' ' * (40-len(t))
        results=t +spa+str(b[i])+'\n'
        f.write(results)
    f.write('\n')
    f.write('_' * 50+'\n')
    f.write('       ### Functions and distributions ###'+'\n')
    f.write('_' * 50+'\n')
    for I in range(num_range_vals):
        multiplier=1
        t=VarNames[properties[I+num_single_vals]].strip()
        if units=='um':
            t=t.replace('px','um')
        f.write('\n')
        f.write('\n')
        f.write('# '+t+'\n')
        f.write('-' * 50+'\n')
        xlabel='Cumulative probability'
        if properties[I+num_single_vals] in [15,16,17]:
            xlabel='Wetting-sat (Sw)'
        if properties[I+num_single_vals] in [18]:
            xlabel='lag (px)'
            multiplier=50
        spa=' ' * (40-len(xlabel))
        f.write(xlabel+spa+'Value'+'\n')
        f.write('-' * 50+'\n')
        shift=I*100+num_single_vals
        for J in range(100):
            t=str(np.round((J*.01+.01)*multiplier,2))
            spa=' ' * (40-len(t))
            f.write(t+spa+str(np.round(vals[J+shift],7))+'\n')
    f.close()
    a=0
    if verbose:
        print('\n')
        with open(FileName,"r") as f:
            for line in f:
                print(line)
                a=a+1
                if a>23:
                    print('-' * 50+'\n')
                    print('To see all the results please refer to this file: \n')
                    print(FileName+'\n')
                    break

# REVISED DEEPORE MODELS
# Reason -> different num of property predictions
# Revision -> new parameter for property num, used instead of 1515
def DeePore1(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    # variable filters / 3 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    return model
def DeePore2(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    # variable filters / 4 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(24, (2, 2), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4)
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    return model
def DeePore3(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    # fixed filter size/ 3 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    return model
def DeePore4(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse'])
    return model
def DeePore5(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(24, (2, 2), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4)
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse'])
    return model
def DeePore6(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse'])
    return model
def DeePore7(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    return model
def DeePore8(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    return model
def DeePore9(INPUT_SHAPE,OUTPUT_SHAPE,property_num):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(18, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)
    c4 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4)
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.leaky_relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=WBCE, metrics=['mse'])
    return model


def DeePoreRevised1(INPUT_SHAPE,OUTPUT_SHAPE,property_num): # varying filters, 4 blocks w/ residuals
    inputs = Input(INPUT_SHAPE[1:])
    # conv block 1
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)

    # conv block 2 with residual
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    residual2 = Conv2D(12,(1,1), kernel_initializer='he_normal', padding='same')(p1)
    c2residual = Add()([c2,residual2])
    p2 = MaxPooling2D((2, 2)) (c2residual)

    # conv block 3 with residual
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    residual3 = Conv2D(18,(1,1), kernel_initializer='he_normal', padding='same')(p2)
    c3residual = Add()([c3,residual3])
    p3 = MaxPooling2D((2, 2)) (c3residual)

    # conv block 4 with residual
    c4 = Conv2D(24, (2, 2), kernel_initializer='he_normal', padding='same') (p3)
    residual4 = Conv2D(24,(1,1), kernel_initializer='he_normal', padding='same')(p3)
    c4residual = Add()([c4,residual4])
    p4 = MaxPooling2D((2, 2)) (c4residual)

    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    return model

def DeePoreRevised2(INPUT_SHAPE,OUTPUT_SHAPE,property_num): # fixed filter, 3 blocks w/ residuals
    inputs = Input(INPUT_SHAPE[1:])
    # conv block 1
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)

    # conv block 2 with residual
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    residual2 = Conv2D(24,(1,1), kernel_initializer='he_normal', padding='same')(p1)
    c2residual = Add()([c2,residual2])
    p2 = MaxPooling2D((2, 2)) (c2residual)

    # conv block 3 with residual
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    residual3 = Conv2D(36,(1,1), kernel_initializer='he_normal', padding='same')(p2)
    c3residual = Add()([c3,residual3])
    p3 = MaxPooling2D((2, 2)) (c3residual)

    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(property_num, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(property_num, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse'])
    return model

# REVISED DEEPORE
# revision -> added routes to new models, skips number 10 but that is intentional so that model type 11=dpr1 and 12=dpr2 (easier to map)
def modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType,property_num=1515):
    if ModelType==1:
        model=DeePore1(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==2:
        model=DeePore2(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==3:
        model=DeePore3(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==4:
        model=DeePore4(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==5:
        model=DeePore5(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==6:
        model=DeePore6(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==7:
        model=DeePore7(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==8:
        model=DeePore8(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==9:
        model=DeePore9(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==11:
        model=DeePoreRevised1(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    if ModelType==12:
        model=DeePoreRevised2(INPUT_SHAPE,OUTPUT_SHAPE,property_num)
    return model

# ORIGINAL DEEPORE
def splitdata(List):
    N=np.int32([0,len(List)*.8,len(List)*.9,len(List)])
    TrainList=List[N[0]:N[1]]
    EvalList=List[N[1]:N[2]]
    TestList=List[N[2]:N[3]]
    return TrainList, EvalList, TestList

# ORIGINAL DEEPORE
def check_get(url,File_Name):
    def download_callback(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d MB / %d MB" % (
                percent, len(str(totalsize)), readsofar/1e6, totalsize/1e6)
            sys.stderr.write(s)
            if readsofar >= totalsize: # near the end
                sys.stderr.write("\n")
        else: # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))
    if not os.path.isfile(File_Name):
        ans=input('You dont have the file "' +File_Name +'". Do you want to download it? (Y/N) ')
        if ans=='Y' or ans=='y' or ans=='yes' or ans=='Yes' or ans=='YES':
            print('Beginning file download. This might take several minutes.')
            urlretrieve(url,File_Name,download_callback)
    else:
        print('File "' +File_Name +'" is detected on your machine.'  )

# ORIGINAL DEEPORE
def shuf(L):
    import random
    random.shuffle(L)
    return L

# ORIGINAL DEEPORE
def WMSE(y_actual,y_pred): # weighted MSE loss
    w=np.ones((1,1515))
    w[:,15:]=.01
    w=np.float32(w)
    w=tf.tile(w, [tf.shape(y_pred)[0],1])
    loss=kb.square(y_actual-y_pred)
    loss=tf.multiply(loss,w)
    return loss

# ORIGINAL DEEPORE
def WBCE(y_actual,y_pred): # weighted binary crossentropy loss
    w=np.ones((1,1515))
    w[:,15:]=.01
    w=np.float32(w)
    w=tf.tile(w, [tf.shape(y_pred)[0],1])
    loss=kb.square(y_actual-y_pred)
    loss=y_actual*(-tf.math.log(y_pred))+(1-y_actual)*(-tf.math.log(1-y_pred))
    loss=tf.multiply(loss,w)
    return loss

# ORIGINAL DEEPORE
def hdf_shapes(Name,Fields):
    # Fields is list of hdf file fields
    Shape = [[] for _ in range(len(Fields))]
    with h5py.File(Name, 'r') as f:
        for I in range(len(Fields)):
            Shape[I]=f[Fields[I]].shape
    return Shape

# ORIGINAL DEEPORE
def now():
    import datetime
    d1 = datetime.datetime(1, 1, 1)
    d2 = datetime.datetime.now()
    d=d2-d1
    dd=d.days+d.seconds/(24*60*60)+d.microseconds/(24*60*60*1e6)+367
    return dd
def nowstr():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d-%b-%Y %H.%M.%S")

# ORIGINAL DEEPORE
def mat2np(Name): # load the MATLAB array as numpy array
    B=sio.loadmat(Name)
    return B['A']

# ORIGINAL DEEPORE
def writeh5slice(A,FileName,FieldName,Shape):
    # example: writeh5slice(A,'test3.h5','X',Shape=[70,70,1])
    D=len(Shape)
    if D==2:
         maxshape=(None,Shape[0],Shape[1])
         Shape0=(1,Shape[0],Shape[1])
         A=np.reshape(A,Shape0)
    if D==3:
         maxshape=(None,Shape[0],Shape[1],Shape[2])
         Shape0=(1,Shape[0],Shape[1],Shape[2])
         A=np.reshape(A,Shape0)
    if D==4:
         maxshape=(None,Shape[0],Shape[1],Shape[2],Shape[3])
         Shape0=(1,Shape[0],Shape[1],Shape[2],Shape[3])
         A=np.reshape(A,Shape0)
    try:
        with h5py.File(FileName, "r") as f:
            arr=f[FieldName]
        with h5py.File(FileName, "a") as f:
            arr=f[FieldName]
            Slice=arr.shape[0]
            arr.resize(arr.shape[0]+1, axis=0)
            arr[Slice,...]=A
        print('writing slice '+ str(Slice))
    except:
        with h5py.File(FileName, "a") as f:
            f.create_dataset(FieldName, Shape0,maxshape=maxshape, chunks=True,dtype=A.dtype,compression="gzip", compression_opts=5)
            f[FieldName][0,...]=A

# ORIGINAL DEEPORE
def normalize(A):
    A_min = np.min(A)
    return (A-A_min)/(np.max(A)-A_min)
def normal(A):
    A_min = np.min(A)
    return (A-A_min)/(np.max(A)-A_min)

# ORIGINAL DEEPORE
def makeblocks(SS,n=None,w=None,ov=0):
    # w is the fixed width of the blocks and n is the number of blocks
    # if the number be high while w is fixed, blocks start to overlap and ov is between 0 to 1 gets desired overlapping degree
    # example:dp.makeblocks([100,200],w=16,ov=.1)
    HI=[]
    LO=[]
    for S in SS:
        if w==None and n!=None:
            mid=np.ceil(np.linspace(0,S,n+1));
            lo=mid; lo=np.delete(lo,-1);
            hi=mid; hi=np.delete(hi,0);
        if w!=None and n!=None:
            mid=np.ceil(np.linspace(0,S-w,n));
            lo=mid;
            hi=mid+w
        if w!=None and n==None: # good for image translation
            mid=np.asarray(np.arange(0,S,int(w*(1-ov))))
            lo=mid;
            hi=mid+w
            p=np.argwhere(hi>S)
            if len(p)>0:
                diff=hi[p]-S
                hi[p]=S
                lo[p]=lo[p]-diff
                hi=np.unique(hi)
                lo=np.unique(lo)
        HI.append(hi)
        LO.append(lo)
    return LO,HI

# ORIGINAL DEEPORE
def readh5slice(FileName,FieldName,Slices):
    with h5py.File(FileName, "r") as f:
         A=f[FieldName][np.sort(Slices),...]
    return A

# ORIGINAL DEEPORE
def parfor(func,values):
    # example
    # def calc(I):
    #     return I*2
    # px.parfor(calc,[1,2,3])
    N=len(values)
    from joblib import Parallel, delayed
    from tqdm import tqdm
    Out = Parallel(n_jobs=-1)(delayed(func)(k) for k in tqdm(range(1,N+1)))
    return Out

