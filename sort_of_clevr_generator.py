import cv2
import os
import numpy as np
import random

import pickle
import warnings
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

train_size = 9800
test_size = 200
img_size = 75
size = 5
question_size = 11  ## 1 x (6 for one-hot vector of color), 2 for question type, 3 for question subtype
q_type_idx = 6
sub_q_type_idx = 8
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]


try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

#pick random centers until new object does not overlap with existing objects
def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

state_row_length = 15

def build_dataset(index, df):
    objects = []
    img = np.ones((img_size, img_size, 3)) * 255

    for color_id, color in enumerate(colors):
        center = center_generate(objects)
        row = np.zeros(state_row_length)
        if random.random() < 0.5:
            start = (center[0] - size, center[1] - size)
            end = (center[0] + size, center[1] + size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id, center, 'r'))
            # state description

            row[0] = index #object ID
            row[1+color_id] = 1 #color
            row[7] = 1  #shape, index 7-8 refers to the shape of object rectangle = 1 0, circle = 0 1
            row[9] = center[0] / 75 #x-coordinate
            row[10] = center[1] / 75 #y-coordinate
            row[11] = 1 #material, index 11-12 refers to material smooth/shiny (all smooth)
            row[13] = 1 #size, index 13-14 refers to size small/big (all small)
            
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id, center, 'c'))
            row[0] = index
            row[1+color_id] = 1
            row[8] = 1 #circle
            row[9] = center[0] / 75 
            row[10] = center[1] / 75
            row[11] = 1 
            row[13] = 1 
        
        df.loc[len(df.index)] = row
    binary_questions = []
    norel_questions = []
    binary_answers = []
    norel_answers = []
    """Non-relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, 0, 1, 2, 3, 4, 5]"""
        if subtype == 0:
            """query shape->rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    """Binary Relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        binary_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """count->1~6"""
            #how many objects have the same shape as my-object
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        #answer is an int  between 0 and 9, needs to be one-hot encoded
        binary_answers.append(answer)

    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    
    img = img/255.
    dataset = (img, binary_relations, norelations)
    return dataset


print('building test datasets...')
COLUMNS = ['img_id', 'obj_id_0', 'obj_id_1', 'obj_id_2', 'obj_id_3', 'obj_id_4', 'obj_id_5', 'shape_r', 'shape_c', 'center_x', 'center_y', 'material_sm', 'material_sh', 'size_s', 'size_b']
scene_description_test = pd.DataFrame(columns=COLUMNS)

test_datasets = [build_dataset(index, scene_description_test) for index in range(test_size)]
print(scene_description_test)
scene_description_train = pd.DataFrame(columns=COLUMNS)
print('building train datasets...')
train_datasets = [build_dataset(index, scene_description_train) for index in range(train_size)]
print(scene_description_train)

scene_description_test.to_csv("data/test_descriptions.csv",index=False)
scene_description_train.to_csv("data/train_descriptions.csv",index=False)

print('saving datasets...')
filename = os.path.join(dirs, 'sort-of-clevr-original.pickle')
with  open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))