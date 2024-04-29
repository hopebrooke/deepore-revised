
# Below is the code used to create the compact dataset on Google Colab
# The functionality for it is defined in the DeePoreRevised file

# Runtime was approximately 5 hours to create the six-slice dataset
# While the dataset generated is created already and available, this code can be run to recreate it and verify the correctness
# Dataset route names, and importing the deeporerevised library will need to be adapted for your repository structure
# Worth making sure that you don't accidentally delete/write over the created dataset provided

#-----------------------------------------------------------------------------------------------------------------------------------

# If using google colab:
# load google drive content
# from google.colab import drive
# drive.mount('/content/drive')
# navigate to project folder
# %cd /content/drive/MyDrive/hb-final-project


import h5py
import DeePoreRevised as dpr

# define original dataset route
full_dataset = '/content/drive/MyDrive/Project/DeePore_Dataset.h5'
# define new dataset route
new_dataset = '/content/drive/MyDrive/Project/DeePoreRevised_Compact_Data.h5'

# create the dataset
dpr.create_compact_dataset(full_dataset, new_dataset, n=2)  # n=2 means two slices from each direction of the material, set n=1 to recreate the original DeePore compact dataset


# validation of created dataset
# open the HDF5 file in read mode
with h5py.File(new_dataset, 'r') as f:

    # print the keys at the root level of the HDF5 file
    print("Root keys:", list(f.keys()))

    # print the shape and dtype of 'X' dataset (the materials)
    dataset_X = f['X']
    print("Shape of dataset X:", dataset_X.shape)
    print("Data type of dataset X:", dataset_X.dtype)

    # print the shape and dtype of 'Y' dataset (the properties)
    dataset_Y = f['Y']
    print("Shape of dataset Y:", dataset_Y.shape)
    print("Data type of dataset Y:", dataset_Y.dtype)