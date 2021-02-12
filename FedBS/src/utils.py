#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal, cifar_noniid_unequal_zipf
import torch.nn.functional as F
from collections import defaultdict
from options import args_parser


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = cifar_noniid_unequal(train_dataset, args.num_users)

            elif args.unequalZipf:
                user_groups = cifar_noniid_unequal_zipf(train_dataset, args.num_users)

            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups



def average_weights_baseline(w):
    """
    Returns the average of the weights.
    """
    BN_LAYERS = []
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if "normalize" not in key:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        else:
            BN_LAYERS.append(key)
    for layer in BN_LAYERS:
        w_avg.pop(layer)
    return w_avg


def average_weights(w, rm_dict, rv_dict, *args):
    """
    Returns the average of the weights.
    """
    
    if args:
        num_data_per_client = args[0] 
        rm, rv = average_bn_statistics_unbal(rm_dict, rv_dict, num_data_per_client) 
    else: 
        rm, rv = average_bn_statistics_bal(rm_dict, rv_dict)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))


    args = args_parser()

    w_avg["layer1.1.running_mean"] = rm[0]
    w_avg["layer2.1.running_mean"] = rm[1]

    if args.norm == "BN":   
        w_avg["layer1.1.running_var"] = rv[0]
        w_avg["layer2.1.running_var"] = rv[1]
    elif args.norm == "BRN":
        w_avg["layer1.1.running_std"] = torch.sqrt(rv[0])
        w_avg["layer2.1.running_std"] = torch.sqrt(rv[1])

    '''
    w_avg["BNorm1.running_mean"] = rm[0]
    w_avg["BNorm2.running_mean"] = rm[1]

    if args.norm == "BN":   
        w_avg["BNorm1.running_var"] = rv[0]
        w_avg["BNorm2.running_var"] = rv[1]
    elif args.norm == "BRN":
        w_avg["BNorm1.running_std"] = torch.sqrt(rv[0])
        w_avg["BNorm2.running_std"] = torch.sqrt(rv[1])
    

    w_avg["normalize1.running_mean"] = rm[0]
    w_avg["normalize2.running_mean"] = rm[1]
    w_avg["normalize3.running_mean"] = rm[2]
    w_avg["normalize4.running_mean"] = rm[3]

    if args.norm == "BN":   
        w_avg["normalize1.running_var"] = rv[0]
        w_avg["normalize2.running_var"] = rv[1]
        w_avg["normalize3.running_var"] = rv[2]
        w_avg["normalize4.running_var"] = rv[3]
    elif args.norm == "BRN":
        w_avg["normalize1.running_std"] = rv[0]
        w_avg["normalize2.running_std"] = rv[1]
        w_avg["normalize3.running_std"] = rv[2]
        w_avg["normalize4.running_std"] = rv[3]

    '''

    return w_avg




def average_bn_statistics_bal(rm_dict, rv_dict):
    global_avg_rm, global_avg_rv = [], []
    if rm_dict and rv_dict:
        d_rm, d_rv = {}, {}
        final_rm, final_rv = [], []
        for key1, key2 in zip(rm_dict, rv_dict):
            if len(rm_dict[key1]) > 1:
                l = []
                rmList = list(zip(*rm_dict[key1]))
                rm_dict_square = [[torch.square(t) for t in L] for L in rm_dict[key1]]
                rmList_square = list(zip(*rm_dict_square))

                for i in range(len(rmList)):
                    l.append(torch.mean(torch.stack(rmList[i]), dim=0))
                d_rm[key1] = l
            else:
                d_rm[key1] = rm_dict[key1][0]
            if len(rv_dict[key2]) > 1:
                l = []
                rvList = list(zip(*rv_dict[key1]))

                for i in range(len(rvList)):
                    l.append((torch.mean(torch.stack(rvList[i]), dim=0) + torch.mean(torch.stack(rmList_square[i]), dim=0)) - torch.square(d_rm[key2][i]))
                    #l.append(torch.mean(torch.stack(rvList[i]), dim=0))
                d_rv[key2] = l
            else:
                d_rv[key2] = rv_dict[key2][0]

        l_rv, l_rm = list(d_rv.values()), list(d_rm.values())
        l_rm_square = [[torch.square(t) for t in L] for L in l_rm]

        final_rm, final_rv, final_rm_square = list(zip(*l_rm)), list(zip(*l_rv)), list(zip(*l_rm_square))

        for i in range(len(final_rm)):
            global_avg_rm.append(torch.mean(torch.stack(final_rm[i]), dim=0))
            global_avg_rv.append((torch.mean(torch.stack(final_rv[i]), dim=0) + torch.mean(torch.stack(final_rm_square[i]), dim=0)) - torch.square(global_avg_rm[i]))

    return global_avg_rm, global_avg_rv



def average_bn_statistics_unbal(rm_dict, rv_dict, num_data_per_client):
    # Variables typres
    # rm_dict = {id_client1: [rm_list1_1, rm_list1_2,...], id_client2, [rm_list2_1, rm_list2_2,...], ......}
    # rm_list = [[100], [150], [200], [500]]

    global_avg_rm, global_avg_rv = [], []
    total_num_data = sum(num_data_per_client.values())

    if rm_dict and rv_dict:
        L_rm, L_rv = [], []
        for key1, key2 in zip(rm_dict, rv_dict):
            if (len(rm_dict[key1]) > 1):
                l = []
                rmList = list(zip(*rm_dict[key1]))
                for i in range(len(rmList)):
                    l.append(torch.mean(torch.stack(rmList[i]), dim=0))
                l = [v * len(rm_dict[key1]) * (num_data_per_client[key1] / total_num_data) for v in l]
                L_rm.append(l)
            else:
                l = [v * (num_data_per_client[key1] / total_num_data) for v in rm_dict[key1][0]]
                # L_rm.append(rm_dict[key1][0])
                L_rm.append(l)

            if (len(rv_dict[key2]) > 1):
                l = []
                rvList = list(zip(*rv_dict[key1]))
                for i in range(len(rvList)):
                    l.append(torch.mean(torch.stack(rvList[i]), dim=0))
                l = [v * len(rv_dict[key2]) * (num_data_per_client[key2] / total_num_data) for v in l]
                L_rv.append(l)
            else:
                l = [v * (num_data_per_client[key2] / total_num_data) for v in rv_dict[key2][0]]
                L_rv.append(l)

        final_rm, final_rv = list(zip(*L_rm)), list(zip(*L_rv))

        for i in range(len(final_rm)):
            global_avg_rm.append(torch.mean(torch.stack(final_rm[i]), dim=0))
            global_avg_rv.append(torch.mean(torch.stack(final_rv[i]), dim=0))

        return global_avg_rm, global_avg_rv


def save_model(epoch, model, optimizer, filepath):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)

def load_model(filepath, args):
    if args.gpu:
        state = torch.load(filepath)
    else:
        state = torch.load(filepath, map_location=torch.device('cpu'))
    return state


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Unequal            : {args.unequal}\n')
    print(f'    Moving Average     : {args.ma}\n')
    print(f'    Normalization     : {args.norm}\n')
    return
