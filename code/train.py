import torch
import torch.nn as nn
from model import CNN
from utils import *
import time


def train(model, grid_resolution, max_dist, epoches, batch_size, threshold, criterion, optim,pro_data_set,lig_data_set,train_dataset,
          valid_dataset=None, rotation=False, name=None, early_stop=False,):
    history_loss = dict()
    history_acc = dict()
    history_loss["val"] = []
    history_loss["train"] = []

    history_acc["val"] = []
    history_acc["train"] = []
    best_acc = 0
    positive_ids, negative_ids = train_dataset

    if valid_dataset is not None:
        positive_ids_valid, negative_ids_valid = valid_dataset

    for epoch in range(epoches):

        epoch_data, train_length = create_epoch_data(positive_ids, negative_ids, pro_data_set, lig_data_set, 'limit')
        print(len(epoch_data))
        if rotation:
            mini_data = miniDataset(epoch_data, train_length, 'training',
                                    transform=transform.Compose([rotation_grid(),  # randomly rotate to augment
                                                                 make_grid(grid_resolution=grid_resolution,
                                                                           max_dist=max_dist),
                                                                 array2tensor(torch.FloatTensor)]),
                                    target_transform=array2tensor(torch.FloatTensor))
        else:
            mini_data = miniDataset(epoch_data, 'training', transform=transform.Compose(
                [make_grid(grid_resolution=grid_resolution, max_dist=max_dist),
                 array2tensor(torch.FloatTensor)]),
                                    target_transform=array2tensor(torch.FloatTensor))

        train_loader = torch.utils.data.DataLoader(dataset=mini_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

        if valid_dataset is not None:
            valid_epoch_data, valid_len = create_epoch_data(positive_ids_valid, negative_ids_valid, pro_data_set,
                                                            lig_data_set, 'limit')
            print('valid_data_length:', len(valid_epoch_data))
            valid_mini_data = miniDataset(valid_epoch_data, valid_len, 'training', transform=transform.Compose(
                [make_grid(grid_resolution=grid_resolution, max_dist=max_dist),
                 array2tensor(torch.FloatTensor)]),
                                          target_transform=array2tensor(torch.FloatTensor))

            valid_loader = torch.utils.data.DataLoader(dataset=valid_mini_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

        print('finish preparing data')

        train_loss = 0.0
        train_acc = 0.0
        len_train = 0
        print('start training_{}_epoch'.format(epoch))
        epoch_start_time = time.time()
        model.train()
        for step, (inp, target) in enumerate(train_loader):
            #             print(inp.shape)
            #             print(target.shape)

            y_pred_po = model(inp)
            loss_ = criterion(y_pred_po, target.view(-1, 1))
            optim.zero_grad()
            loss_.backward()
            optim.step()
            train_loss += loss_.item()

            y_pred_po_result = (y_pred_po.detach().numpy() > threshold).reshape(-1)
            #             print("y_pred_po_result shape",y_pred_po_result.shape)
            train_acc += np.sum(y_pred_po_result == target.detach().numpy())
            len_train += target.shape[0]
            #             train_acc+=np.sum(y_pred_neg_result==y_true_neg.detach().numpy())
            if step % 10 == 0:
                print('finish {} iterations'.format(step))

        train_loss = np.sum(train_loss) / len_train
        train_acc = np.sum(train_acc) / len_train
        history_loss["train"].append(train_loss)
        history_acc["train"].append(train_acc)
        print('_' * 20)

        if valid_dataset is not None:
            print("start validation in {} epoch.".format(epoch))
            # eval
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            len_val = 0
            with torch.no_grad():
                for step, (inp, target) in enumerate(valid_loader):
                    y_pred_po = model(inp)
                    #                     print(y_pred_po.shape)
                    #                     print(type(y_pred_po))
                    loss_ = criterion(y_pred_po, target.view(-1, 1))
                    val_loss += loss_.item()
                    y_pred_po_result = (y_pred_po.detach().numpy() > threshold).reshape(-1)
                    val_acc += np.sum(y_pred_po_result == target.detach().numpy())
                    len_val += target.shape[0]
                    if step % 10 == 0:
                        print('finish {} iterations'.format(step))

                val_loss = np.sum(val_loss) / len_val
                val_acc = np.sum(val_acc) / len_val
                history_loss["val"].append(val_loss)
                history_acc["val"].append(val_acc)

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, epoches, time.time() - epoch_start_time, \
                   train_acc, train_loss, val_acc, val_loss))
        else:
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
                  (epoch + 1, epoches, time.time() - epoch_start_time, \
                   train_acc, train_loss))

        if name == None:
            PATH = './model/model_epoch_{}'.format(epoch)
            PATH_E='./model/model_checkpoint'
        else:
            PATH = './model/{}_epoch_{}'.format(name, epoch)
            PATH_E = './model/{}_model_checkpoint'.format(name)

        # self-implement early stop
        if early_stop == True:
            if val_acc > best_acc:
                best_acc = val_acc
                es = 0
                torch.save(model.state_dict(), PATH_E)
                print('Model saved at {}'.format(PATH_E))
            else:
                es += 1
                print("Counter {} of 5".format(es))
            if es == 5:
                print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
                break
        else:
            if epoch % 2 == 0:
                torch.save(model.state_dict(), PATH)
                print('Model saved at {}'.format(PATH))

    return history_loss, history_acc

