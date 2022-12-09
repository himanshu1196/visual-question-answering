import os

import pickle
import random
import numpy as np
import pandas as pd

from PIL import Image

def getDataSamples(datatype = 'train', n = 1):
    dirs = 'data'
    filename = os.path.join('.', dirs,'sort-of-clevr-original.pickle')
    train_filename = os.path.join('.', dirs,'train_descriptions.csv')
    test_filename = os.path.join('.', dirs, 'test_descriptions.csv')
    
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    
    traindf = pd.read_csv(train_filename)
    testdf = pd.read_csv(test_filename)

    print('processing data...')
    rel = []
    norel = []
    if datatype == 'train':
        samples = random.sample(list(range(len(train_datasets))), n)
        

        for index in samples:
            datatuple = train_datasets[index]
            img, relations, norelations = datatuple[0],datatuple[1],datatuple[2]
            state_desc = traindf.loc[traindf['img_id'] == index, :].values[:,1:]
            for qst,ans in zip(relations[0], relations[1]):
                rel.append((img,state_desc,qst,ans))
            for qst,ans in zip(norelations[0], norelations[1]):
                norel.append((img,state_desc,qst,ans))
    
    if datatype == 'test':
        samples = random.sample(list(range(len(test_datasets))), n)
        rel_test = []
        norel_test = []

        for index in enumerate(samples):
            datatuple = test_datasets[index]
            img, relations, norelations = datatuple[0],datatuple[1],datatuple[2]
            state_desc = testdf.loc[testdf['img_id'] == index, :].values[:,1:]
            for qst,ans in zip(relations[0], relations[1]):
                rel.append((img,state_desc,qst,ans))
            for qst,ans in zip(norelations[0], norelations[1]):
                norel.append((img,state_desc,qst,ans))

    return (rel, norel)

def save_img_sample():
    rel, norel = getDataSamples()
    img = Image.fromarray((np.array(rel[0][0]) * 255.).astype(np.uint8), "RGB")
    img.save(os.path.join('E:\\NYU\\Fall 22\\Artificial Intelligence 1\\vqa', 'train_sample.png'))
    print('State description : ')
    print(rel[0][1])
    print('Question : ')
    print(rel[0][2])
    print('Answer : ')
    print(rel[0][3])

save_img_sample()