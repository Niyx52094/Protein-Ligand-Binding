import argparse
import numpy as np
import pickle
from utils import *
import random
from model import CNN
from train import *
from eval_function import *

def main():
    parser = argparse.ArgumentParser(description="Run bs6207 project.")
    parser.add_argument('-lr', type=float, dest='lr', help='learning rate')
    parser.add_argument('-dropout', type=float, dest='dropout', help='dropout rate')
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose of this function')
    parser.add_argument('-Mybest', type=int, dest='Mybest', help='use best model I already trained or the model you train for evaluation')

    A = parser.parse_args()

    if A.purpose=='preprocess':
        print("wait for preprocessing")
        ## generate trainning data
        str_num = []
        for i in range(1, 3001):
            index = (4 - len(str(i))) * '0' + str(i)
            str_num.append(index)

        pos_data_set = []
        neg_data_set = []
        for i in str_num:
            for j in str_num:
                if i == j:
                    pos_data_set.append((i, j))
                else:
                    neg_data_set.append((i, j))

        lig_data_set = dict()
        for i in str_num:
            path = "./data/training_data/{}_lig_cg.pdb".format(i)
            X, Y, Z, atom_types = read_pdb(path)
            one_image = []
            for x, y, z, atom_type in zip(X, Y, Z, atom_types):
                if atom_type == 'p':
                    pixel = np.array([x, y, z, 1, 0, -1])
                else:
                    pixel = np.array([x, y, z, 0, 1, -1])
                one_image.append(pixel)
            lig_data_set[i] = np.stack(one_image, axis=0)

        # protein dataset
        pro_data_set = dict()
        for i in str_num:
            path = "./data/training_data/{}_pro_cg.pdb".format(i)
            X, Y, Z, atom_types = read_pdb(path)
            one_image = []
            for x, y, z, atom_type in zip(X, Y, Z, atom_types):
                if atom_type == 'p':
                    pixel = np.array([x, y, z, 1, 0, 1])
                else:
                    pixel = np.array([x, y, z, 0, 1, 1])
                one_image.append(pixel)
            pro_data_set[i] = np.stack(one_image, axis=0)

        path='./data/pro_data_set.pkl'
        save_pickle(pro_data_set,path)

        path='./data/lig_data_set.pkl'
        save_pickle(lig_data_set,path)

        random.shuffle(pos_data_set)
        random.shuffle(neg_data_set)

        train_pos_data = pos_data_set[:int(0.8 * len(pos_data_set))]
        train_neg_data = neg_data_set[:int(0.8 * len(neg_data_set))]

        valid_pos_data = pos_data_set[int(0.8 * len(pos_data_set)):]
        valid_neg_data = neg_data_set[int(0.8 * len(pos_data_set)):]

        path='./data/train_pos_data.pkl'
        save_pickle(train_pos_data,path)

        path='./data/train_neg_data.pkl'
        save_pickle(train_neg_data,path)

        path='./data/valid_pos_data.pkl'
        save_pickle(valid_pos_data,path)

        path='./data/valid_neg_data.pkl'
        save_pickle(valid_neg_data,path)

        print("trainning data done")

        ###############################################

        ## generate test data
        str_num_test = []
        for i in range(1, 825):
            index = (4 - len(str(i))) * '0' + str(i)
            str_num_test.append(index)

        test_data_set = dict()
        for i in str_num_test:
            test_data_set[i] = str_num_test

        path='./data/test_data_set.pkl'
        save_pickle(test_data_set,path)

        # lig-dataset
        lig_test_set = dict()
        for i in str_num_test:
            path = './data/testing_data/{}_lig_cg.pdb'.format(i)
            X, Y, Z, atom_types = read_pdb_test(path)
            #     print(atom_types)
            one_image = []
            for x, y, z, atom_type in zip(X, Y, Z, atom_types):
                if atom_type == 'p':
                    pixel = np.array([x, y, z, 1, 0, -1])
                else:
                    pixel = np.array([x, y, z, 0, 1, -1])
                one_image.append(pixel)
            lig_test_set[i] = np.stack(one_image, axis=0)

        # pro-dataset
        pro_test_set = dict()
        for i in str_num_test:
            path = './data/testing_data/{}_pro_cg.pdb'.format(i)
            X, Y, Z, atom_types = read_pdb_test(path)
            #     print(atom_types)
            one_image = []
            for x, y, z, atom_type in zip(X, Y, Z, atom_types):
                if atom_type == 'p':
                    pixel = np.array([x, y, z, 1, 0, 1])
                else:
                    pixel = np.array([x, y, z, 0, 1, 1])
                one_image.append(pixel)
            pro_test_set[i] = np.stack(one_image, axis=0)

        path='./data/lig_test_set.pkl'
        save_pickle(lig_test_set,path)

        path='./data/pro_test_set.pkl'
        save_pickle(pro_test_set,path)

        print("test data done")
        #############################################
        str_num_imitation = []

        for i in range(1, 3001):
            index = (4 - len(str(i))) * '0' + str(i)
            str_num_imitation.append(index)

        imitation_data_set = dict()
        for i in valid_pos_data:
            ii = i[0]
            samples = random.sample(str_num_imitation, 824)
            #     print(type(samples))
            #     print(len(samples))
            #     print(samples)
            if ii in samples:
                total_samples = samples
                print('contain positive')
            else:
                total_samples = random.sample(samples, 823) + [ii]
                print('no positive')
            imitation_data_set[ii] = total_samples
        #     imitation_data_set[i]=str_num_test
        for k, v in imitation_data_set.items():
            imitation_data_set[k] = np.sort(v)

        path='./data/imitation_data_set.pkl'
        save_pickle(imitation_data_set,path)

        print("preprocessing data done")


    elif A.purpose=='train':

        path='./data/pro_data_set.pkl'
        pro_data_set=load_pickle(path)

        path='./data/lig_data_set.pkl'
        lig_data_set=load_pickle(path)

        path='./data/train_pos_data.pkl'
        train_pos_data=load_pickle(path)

        path='./data/train_neg_data.pkl'
        train_neg_data=load_pickle(path)

        path='./data/valid_pos_data.pkl'
        valid_pos_data=load_pickle(path)

        path='./data/valid_neg_data.pkl'
        valid_neg_data=load_pickle(path)
        print('preparing data done')

        model_=CNN(A.dropout)

        optim=torch.optim.Adam(model_.parameters(),lr=A.lr)

        loss=nn.BCEWithLogitsLoss()

        train_dataset = [train_pos_data, train_neg_data]
        valid_dataset = [valid_pos_data, valid_neg_data]
        print("start to train")
        train(model_,1,10,50,64,0,loss,optim,pro_data_set,lig_data_set,train_dataset,valid_dataset,True,'final',True)


    elif A.purpose=='output':

        path='./data/test_data_set.pkl'
        test_data_set=load_pickle(path)

        final_model=CNN(A.dropout)
        if A.Mybest==1:
            final_model_path='./model/Mybest_model.txt'
        else:
            final_model_path = './model/final_model_checkpoint'

        final_model.load_state_dict(torch.load(final_model_path))
        print("load final model done")

        path='./data/lig_test_set.pkl'
        lig_test_set=load_pickle(path)

        path='./data/pro_test_set.pkl'
        pro_test_set=load_pickle(path)



        result = get_rank(final_model, test_data_set, pro_test_set, lig_test_set)
        result=change_str_to_int(result)

        file_name=r'./test_predictions.txt'
        export_dict(result, file_name)

        print('output ligand files done')

    elif A.purpose=='evaluate':

        path='./data/pro_data_set.pkl'
        pro_data_set=load_pickle(path)

        path='./data/lig_data_set.pkl'
        lig_data_set=load_pickle(path)


        path='./data/imitation_data_set.pkl'
        imitation_data_set=load_pickle(path)

        final_model=CNN(A.dropout)
        if A.Mybest==1:
            final_model_path='./model/Mybest_model.txt'
        else:
            final_model_path = './model/final_model_checkpoint'

        final_model.load_state_dict(torch.load(final_model_path))
        print("load final model done")

        imitation_result = get_rank(final_model, imitation_data_set, pro_data_set, lig_data_set)

        sr_=cal_sr(imitation_result)

        ndcg=cal_NDCG(imitation_result)

        print("the total success rate that the target exist in top 10 is:{}, and the nDCG is {}".format(sr_,ndcg))
    else:
        print("please import the right purpose")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
