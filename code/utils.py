
import math
from math import pi
# from itertools import combinations
import torch.utils.data as data
import math
import random
import torch
from torch import nn
import torchvision.transforms as transform
import numpy as np
import os
import pickle
from collections import Counter
import seaborn as sns
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay,auc,confusion_matrix,classification_report,roc_auc_score,f1_score
import pandas as pd
import matplotlib.pyplot as plt


class rotation_grid():
    '''
    rotate the grid with a specific rotation axis
    '''

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, sample):
        pro, lig = sample
        if self.axis is None:
            self.axis = np.random.uniform(size=(3,))
        else:
            self.axis = np.asarray(self.axis)

        theta = np.random.rand() * 2 * math.pi
        self.axis = self.axis / np.sqrt(np.dot(self.axis, self.axis))
        a = np.cos(theta / 2.0)
        b, c, d = -self.axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        new_axis_pro = np.dot(rot, pro[:, :3].T)
        new_axis_lig = np.dot(rot, lig[:, :3].T)
        return (np.hstack((new_axis_pro.T, pro[:, 3:])), np.hstack((new_axis_lig.T, lig[:, 3:])))


class make_grid():
    '''
    make grids of ligand and protein and stack them together in the C(channel) dimension
    '''
    def __init__(self, grid_resolution=1.0, max_dist=10.0):
        self.grid_resolution = grid_resolution
        self.max_dist = max_dist

    def __call__(self, sample):
        pro, lig = sample
        #         print('pro',pro)
        #         print("_"*10)
        center = cal_center(lig)
        grid_pro = change(pro, self.max_dist, self.grid_resolution, center)
        grid_lig = change(lig, self.max_dist, self.grid_resolution, center)
        pro_lig = np.concatenate((grid_pro, grid_lig), axis=0)
        #         print("pro_lig shape is ",pro_lig.shape)

        return pro_lig


class miniDataset(data.Dataset):
    '''
    generate a Dataset inherit Dataset functions
    '''
    def __init__(self, dataset, length, mode='training', format='array', transform=None, target_transform=None):
        self.dataset = dataset
        len_total = len(dataset)
        print(len_total)
        self.target = np.array([1] * length + [0] * (len_total - length))
        #         print(len(self.target))
        #         print(self.target[599])
        #         print(self.target[600])
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.format = format

    def __getitem__(self, index):
        pro, lig = self.dataset[index][0], self.dataset[index][1]
        target = np.array(self.target[index])
        #         print('dataset target:',target)

        if self.transform is not None:
            inp = self.transform((pro, lig))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return inp, target

    def __len__(self):
        return len(self.dataset)

def read_pdb(filename):
    """
    this function used to read train files

    """

    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def read_pdb_test(filename):
    '''
    read test files
    '''
    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        splitted_line = stripped_line.split('\t')
        #         print(splitted_line)
        #         print(type(splitted_line[0]))

        #         print(splitted_line[0])
        #         print(str(splitted_line[0]).strip())
        #         print(splitted_line)
        #         print(str(splitted_line[3]))

        X_list.append(float(str(splitted_line[0].strip())))
        Y_list.append(float(str(splitted_line[1].strip())))
        Z_list.append(float(str(splitted_line[2].strip())))

        atomtype = str(splitted_line[3])
        atomtype_list.append(atomtype)

    return X_list, Y_list, Z_list, atomtype_list

def cal_center(data_array):
    """calcuate the center coordinates of ligand"""
    coord = data_array[:, :3]
    #     print(np.mean(coord,axis=0))
    return np.mean(coord, axis=0)


def change(data_set, max_dist, grid_resolution, center):
    """

    :param data_set: data set contains the corordiates and features to describe atoms
    :param max_dist: the distance compared with the block edge and the center
    :param grid_resolution: how dense the block will be
    :param center: block center coordinates
    :return: the processed 4D grid with XYZ as coordinates and dimension C as the features
    """
    coords = data_set[:, :3]
    features = data_set[:, 3:]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)
    # transform coordinate into the coordinate whose center is the center we set instead of (0,0,0)

    coords = coords - center
    #             print(coords)
    box_size = math.ceil(2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = np.round(grid_coords).astype(int)
    #     print(grid_coords)
    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    #     print("check")
    #     print(features[in_box])
    grid = np.zeros((box_size, box_size, box_size, features.shape[1]))
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[x, y, z] += f
    grid = grid.swapaxes(0, -1)  # C,X change
    grid = grid.swapaxes(1, -1)  # Y,X change
    grid = grid.swapaxes(2, -1)  # ，Z，Y changge,    #final grid shape(C,X,Y,Z)

    #     print('grid shape',grid.shape)
    return grid

def create_epoch_data(train_pos_data, train_neg_data=None, pro_data_set=None, lig_data_set=None, strategy='limit'):
    '''
    :param train_pos_data: positive pairs (pro id,ligand id) dataset
    :param train_neg_data: negative pairs(pro id, ligand id) dataset
    :param pro_data_set: the dict contains every features of a protein
    :param lig_data_set:
    :param strategy: use all the negative data or just the same size of positive data
    :return: the data contains the positive and negative data, and the length of the positive data to make labels.
    '''
    train_pos_data_set = []
    for item in train_pos_data:
        #         print(pro_data_set[item[0]])
        #         print(lig_data_set[item[1]])
        train_pos_data_set.append((pro_data_set[item[0]], lig_data_set[item[1]]))

    data_set__ = train_pos_data_set
    length = len(data_set__)
    if train_neg_data is not None:
        train_neg_data_set = []
        random.shuffle(train_neg_data)
        if strategy == 'limit':
            train_neg_data__ = train_neg_data[:len(train_pos_data)]

            for item in train_neg_data__:
                train_neg_data_set.append((pro_data_set[item[0]], lig_data_set[item[1]]))
        elif strategy == 'all':
            for item in train_neg_data:
                train_neg_data_set.append((pro_data_set[item[0]], lig_data_set[item[1]]))
        else:
            print("Wrong strategy!")

        data_set__ = data_set__ + train_neg_data_set
    return data_set__, length


class array2tensor():
    """Convert ndarrays in sample to Tensors. Samples are assumed to be python dics"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, sample):
        return torch.from_numpy(sample).type(self.dtype)


def test_for_rank(model, batch_size, dataset, length):
    '''

    :param model:
    :param batch_size:
    :param dataset:
    :param length:
    :return:
    '''
    model.eval()
    y_pred = []
    print(length)
    print(len(dataset))
    valid_mini_data = miniDataset(dataset, length, 'training', transform=transform.Compose([make_grid(),
                                                                                            array2tensor(
                                                                                                torch.FloatTensor)]),
                                  target_transform=array2tensor(torch.FloatTensor))

    valid_loader = torch.utils.data.DataLoader(dataset=valid_mini_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)
    with torch.no_grad():
        for step, (inp, target) in enumerate(valid_loader):
            y_pred_po = model(inp)
            #             print(y_pred_po.reshape(-1).shape)
            #                     print(type(y_pred_po))
            y_pred_po = torch.sigmoid(y_pred_po.squeeze())
            y_pred.append(y_pred_po.numpy())
            if step % 100 == 0:
                print('finish {} iterations'.format(step))
        y_pred = np.array(y_pred)
        index = np.argsort(y_pred)

        #         index+=1
        top10 = index[-10:]
        #         print(y_pred[top10[-1]-1])
        return list(reversed(top10))


def get_rank(model, test_data_set, pro_test_set, lig_test_set):
    '''
    get the test files.

    '''
    result = dict()
    step_ = 0
    for k, v in test_data_set.items():
        li = []
        for vv in v:
            li.append((k, vv))
        print('length', len(li))
        data_set, length = create_epoch_data(li, None, pro_test_set, lig_test_set, strategy='limit')
        top10 = test_for_rank(model, 1, data_set, length)
        top10 = np.array(v)[top10]
        result[k] = list(top10)
        print(step_)
        step_ += 1
    return result

def change_str_to_int(result):
    '''
    change the result files value from string to integer

    '''
    for k, v in result.items():
        vv=[]
        for vvv in v:
            vv.append(int(vvv))
        result[k]=vv
    return result

#prediction label compared with test label, return the classification_report.
def modelPerformance(y_test,y_pred,thres):
    cm=confusion_matrix(y_test,y_pred)
    df_cm=pd.DataFrame(cm)
    sns.set(font_scale=2.0)
    sns.heatmap(df_cm, annot=True,fmt='g',cmap ='Blues')
    plt.title("threshold="+str(round(thres,4)))
    print(classification_report(y_test,y_pred))


def export_dict(dict_files,path):
    with open(path,'w') as f:
        f.write('pro_id'+'\t')
        for i in range(1,11):
            name='lig{}_id'.format(i)
            if i!=10:
                f.write(name+'\t')
            else:
                f.write(name+'\n')
        for i in range(1,825):
            pro_name=(4-len(str(i)))*'0'+str(i)
            f.write(str(i)+'\t')
            for j in range(len(dict_files[pro_name])):
                num=dict_files[pro_name][j]
                if(j==9):
                    f.write(str(num)+'\n')
                else:
                    f.write(str(num)+'\t')


def save_pickle(a, path):
    with open(path, 'wb') as f:
        pickle.dump(a, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)