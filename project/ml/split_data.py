'''
dataset_split.py
This file splits a dataset into training
and test folders. The training data gets
split into validation partitions during
training.
'''

import os
from sklearn.model_selection import train_test_split
import shutil

'''
This function will split a folder which contains
subfolders of classes into train and test folders
with corresponding class subfolders.
Parameters:
  folder_path - the path to the folder with the data
  
  test_percent - the desired percent of the data used for testing.
                  Train data will be (1 - test_percent)
'''
def split_dataset(folder_path,test_percent):

    # get partitions for train and test
    train_files, test_files = split_folder(folder_path, test_percent)
    
    # create the train and test folders
    os.mkdir(os.path.join(folder_path,"train"))
    os.mkdir(os.path.join(folder_path,"test"))
    
    # copy training files to training directory
    for img_file in train_files:
        shutil.copy(os.path.join(folder_path,img_file),os.path.join(folder_path,"train",img_file))
    
    # copy test files to test directory
    for img_file in test_files:
        # print(os.path.join(folder_path,img_file))
        shutil.copy(os.path.join(folder_path,img_file),os.path.join(folder_path,"test",img_file))
        

'''
This function will split a folder into training
and test data given a testing percent. This assumes
the folder contains a single class of data.
Parameters:
  folder_path - the path to the folder with the images
  
  test_percent - the desired percent of the data used for testing.
                  Train data will be (1 - test_percent)
'''
def split_folder(folder_path,test_percent):
    # get the file names
    imgs = os.listdir(folder_path)
    train,test = train_test_split(imgs,test_size=test_percent)
    return train,test


if __name__ == '__main__':
    split_dataset("palm_imgs\\data",0.1)