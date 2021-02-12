#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import logging

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, save_model, load_model
from Models import Resnet_cifar10 as resc
from Models import fixup_resnet_cifar as fixup
from Models.GroupNorm import ResnetGN_Cifar10 as ResGN
from Models.BatchRenorm import CNN_BRN as cnn_brn
from Models.BatchRenorm import ResnetBRN_Cifar10 as ResnetBRN
from collections import defaultdict



if __name__ == '__main__':
    start_time = time.time()
    np.random.seed(0)

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.norm == "BN":
        if args.model == 'cnn':
            if args.dataset == 'mnist':
                global_model = CNNMnist()
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist()
            elif args.dataset == 'cifar':
                global_model = CNNCifar()

        elif args.model == 'resnet':
            if args.dataset == 'cifar':
                global_model = resc.ResNet18()
        
        elif args.model == 'fixup':
            if args.dataset == 'cifar':
                global_model = fixup.fixup_resnet18()
        
        elif args.model == 'ResGN':
            if args.dataset == 'mnist':
                global_model = ResGN.ResNet18()
            elif args.dataset == 'cifar':
                global_model = ResGN.ResNet18()

        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')


    elif args.norm == 'BRN':

        if args.model == 'cnn':
            if args.dataset == 'cifar':
                global_model = cnn_brn.CNNCifar_BRN()
                #global_model = ResnetBRN.ResNet18()
            if args.dataset == 'mnist':
                global_model = cnn_brn.CNNMnist_BRN()
            if args.dataset == 'fmnist':
                global_model = cnn_brn.CNNFMnist_BRN()
        else:
            exit('Error: unrecognized model')
    
    cwd = os.getcwd()

    filepath = os.path.join(cwd,'save/objects/{}_{}_C[{}]_iid[{}]_B[{}]_N[{}]_MA[{}]_R[{}].pkl'.\
    format(args.dataset, args.model, args.frac, args.iid,
    args.local_bs, args.norm, args.ma, args.run))

    # Logging setup
    filename = os.path.join(cwd,'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_MA[{}]_R[{}].log'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs, args.norm, args.ma, args.run))
    
    logging.basicConfig(filename=filename, filemode='a', level=logging.INFO)
   

    if args.resume == 1:
        state = load_model(filepath, args)
        global_model.load_state_dict(state['state_dict'])
        #args.optimizer.load_state_dict(state['optimizer'])
        prev_epoch = state['epoch']
    else:
        prev_epoch = 0
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights

    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 50
    save_every = 100
    save_log = 1
    val_loss_pre, counter = 0, 0

    num_data_per_client = {}
    rm_dict, rv_dict = defaultdict(list), defaultdict(list)

    

    for epoch in tqdm(range(prev_epoch, args.epochs+prev_epoch)):

        local_weights, local_losses, local_accuracies= [], [], []
        if (epoch+1) % print_every == 0:
            print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        num_data_per_client.update((key, len(value)) for key, value in user_groups.items() if key in idxs_users)

        for idx in idxs_users:
            rm_list, rv_list = [], []
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, accuracy, optimizer = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            w_copy = copy.deepcopy(w)

            for key in w_copy.keys():
                if "mean" in key:
                    rm_list.append(w.pop(key))
                elif "var" in key:
                    rv_list.append(w.pop(key))
                elif "std" in key:
                    rv_list.append(torch.square(w.pop(key)))

            rm_dict[idx].append(rm_list)
            rv_dict[idx].append(rv_list)


            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_accuracies.append(copy.deepcopy(accuracy))
        
        # Saving the objects train_loss and train_accuracy:
        if (epoch+1) % save_every == 0:
           
            save_model(epoch+1, global_model, optimizer, filepath)
            



        # update global weights
        if args.unequal == 1:
            global_weights = average_weights(local_weights, rm_dict, rv_dict, num_data_per_client)
        else:
            global_weights = average_weights(local_weights, rm_dict, rv_dict)


        # update global weights
        global_model.load_state_dict(global_weights)


        loss_avg = sum(local_losses) / len(local_losses)
        accuracy_avg = sum(local_accuracies)/len(local_accuracies)
        train_loss.append(loss_avg)
        train_accuracy.append(accuracy_avg)


        # print global training loss after every 'i' rounds
        if (epoch+1) % save_log == 0:

            # Test inference
            test_acc, test_l = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            test_loss.append(test_l)

            logging.info('Epoch : %d', epoch+1)
            logging.info('|---- Avg Train Accuracy: {:.2f}'.format(100*np.mean(np.array(train_accuracy))))
            logging.info('|---- Training Loss : {:.2f}'.format(np.mean(np.array(train_loss))))
            logging.info('|---- Test Accuracy: {:.2f}'.format(100*test_acc))
            logging.info('|---- Test loss: {:.2f}'.format(test_l))

            if (epoch+1) % print_every == 0:
                print(f' \n Results after {epoch+1} global rounds of training:')
                #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*np.mean(np.array(train_accuracy))))
                print("|---- Training Loss : ", np.mean(np.array(train_loss)))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
                print("|---- Test loss: ", test_l)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



    # PLOTTING
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Training Loss curve
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(cwd,'save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_MA[{}]_R[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.norm, args.ma, args.run)))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(cwd,'save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_MA[{}]_R[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.norm, args.ma, args.run)))
    
    # Plot test Accuracy and loss vs Communication rounds
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='k')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(cwd,'save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_MA[{}]_R[{}]_test_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.norm, args.ma, args.run)))

    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Test Loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('Testing Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(cwd,'save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_N[{}]_MA[{}]_R[{}]_test_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.norm, args.ma, args.run)))
