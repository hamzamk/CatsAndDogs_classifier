import os
import shutil
import random

def data_splitter(path):
# Create folders and distribute data automatically
# This can easily be a function but there is not enough time, regardless i know there are no human errors
    print('your kaggle dataset of cats and dogs is in', os.getcwd(),  "/dataset/kaggle"')
    os.chdir(path)
    print(os.getcwd()) #sanity check
    if os.path.isdir('train/dogs') is False:
        os.makedirs('train/dogs')
        os.makedirs('train/cats')
        os.makedirs('valid/dogs')
        os.makedirs('valid/cats')
        os.makedirs('test/dogs')
        os.makedirs('test/cats')
    os.chdir('kaggle_data') 
    print(os.getcwd()) #sanity check

    ### this should be a function possibly which takes **kwargs:
    #{class0: 'dog', class1: 'cat', folders:['train, valid, test], num:[5000, 1000, 1000]'}
    #Big O notation : N,  avoided nested loops
    for c in random.sample(glob.glob('cat*'), 5000):
        shutil.move(c, '../train/cats')
    for c in random.sample(glob.glob('dog*'), 5000):
        shutil.move(c, '../train/dogs')
    for c in random.sample(glob.glob('cat*'), 1000):
        shutil.move(c, '../valid/cats')
    for c in random.sample(glob.glob('dog*'), 1000):
        shutil.move(c, '../valid/dogs')
    for c in random.sample(glob.glob('cat*'), 1000):
        shutil.move(c, '../test/cats')
    for c in random.sample(glob.glob('dog*'), 1000):
        shutil.move(c, '../test/dogs')
    os.chdir('../../../')
    print(os.getcwd()) #sanity check