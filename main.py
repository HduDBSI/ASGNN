#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

from __future__ import division
import numpy as np
from model import *
from util import build_graph, Data, split_validation
import pickle
import argparse
import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True
os.environ['KMP_WARNINGS'] = 'off'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla', help='dataset name')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=80, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_false', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--appear_time', type=int, default=5, help='item appear time number')
opt = parser.parse_args()
train_data = pickle.load(open('./data/'+ opt.dataset + '/' + opt.dataset + str(opt.appear_time)  + 'gnnas_train.txt', 'rb'))
test_data = pickle.load(open('./data/'+ opt.dataset + '/' + opt.dataset + str(opt.appear_time) + 'gnnas_test.txt', 'rb'))

n_user = 0
n_node = 0
if opt.dataset == 'gowalla':
    n_node = 24105
    n_user = 6536
elif opt.dataset == 'foursquare':
    n_node = 5515
    n_user = 1786
else:
    n_node = 10905
    n_user = 7580
# g = build_graph(all_train_seq)
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)
model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
             n_user=n_user,
             lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
             lr_dc=opt.lr_dc,
             nonhybrid=opt.nonhybrid)
print(opt)
best_result = [0, 0]
best_epoch = [0, 0]
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt, model.loss_train, model.global_step]
    print('start training: ', datetime.datetime.now())
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets, user, pre, pre_mask, pre_alias = train_data.get_slice(i)
        _, loss, _ = model.run(fetches, targets, item, user, pre, adj_in, adj_out, alias, mask, pre_alias, pre_mask)
        loss_.append(loss)
    loss = np.mean(loss_)

    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [], []
    right5 = 0
    right10 = 0
    right15 = 0
    right20 = 0
    mrr5, mrr10, mrr15, mrr20 = 0, 0, 0, 0
    for i, j in zip(slices, np.arange(len(slices))):
        # adj_in, adj_out, alias, item, mask, targets, user = test_data.get_slice(i)
        adj_in, adj_out, alias, item, mask, targets, user, pre, pre_mask, pre_alias = test_data.get_slice(i)
        # scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, user, adj_in, adj_out, alias, mask)
        scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, user, pre, adj_in, adj_out, alias, mask, pre_alias, pre_mask)

        test_loss_.append(test_loss)
        index20 = np.argsort(scores, 1)[:, -20:]
        index15 = np.argsort(scores, 1)[:, -15:]
        index10 = np.argsort(scores, 1)[:, -10:]
        index5 = np.argsort(scores, 1)[:, -5:]

        for score, target in zip(index5, targets):
            if (target - 1) in score:
                right5 += 1
            if len(np.where(score == target - 1)[0]) == 0:
                mrr5 += 0
            else:
                mrr5 += 1 / (5 - np.where(score == target - 1)[0][0])
        for score, target in zip(index10, targets):
            if (target - 1) in score:
                right10 += 1
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10 += 0
            else:
                mrr10 += 1 / (10 - np.where(score == target - 1)[0][0])
        for score, target in zip(index15, targets):
            if (target - 1) in score:
                right15 += 1
            if len(np.where(score == target - 1)[0]) == 0:
                mrr15 += 0
            else:
                mrr15 += 1 / (15 - np.where(score == target - 1)[0][0])
        for score, target in zip(index20, targets):
            if (target - 1) in score:
                right20 += 1
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20 += 0
            else:
                mrr20 += 1 / (20 - np.where(score == target - 1)[0][0])

    R_5 = float(right5 / test_data.length)
    MRR_5 = float(mrr5 / test_data.length)
    R_10 = float(right10 / test_data.length)
    MRR_10 = float(mrr10 / test_data.length)
    R_15 = float(right15 / test_data.length)
    MRR_15 = float(mrr15 / test_data.length)
    R_20 = float(right20 / test_data.length)
    MRR_20 = float(mrr20 / test_data.length)
    print('R@5' + ' = ' + str(R_5))
    print('MRR@5' + ' = ' + str(MRR_5))
    print('R@10' + ' = ' + str(R_10))
    print('MRR@10' + ' = ' + str(MRR_10))
    print('R@15' + ' = ' + str(R_15))
    print('MRR@15' + ' = ' + str(MRR_15))
    print('R@20' + ' = ' + str(R_20))
    print('MRR@20' + ' = ' + str(MRR_20))

