#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple


query_name_dict = {(('e', ('r', 'r')), ('e', ('r', 'n'))): '1order',
                   ((('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n')))): '2order',
                   ((('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n')))): '3order',
                   }


name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())





def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing LogiRec',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default="data/PW/1-order", help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=1, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=512, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=60.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0002, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=8, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=30000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--save_checkpoint_steps', default=5000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=1000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=500, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--tasks', default='relation.1order.2order.3order', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="[1600,2]", type=str,
                        help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('--checkpoint_name', default=None, type=str, help='name for loading the checkpoints')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args, step):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint' + str(step))
    )


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def evaluate(model, answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    ac_average_metrics = defaultdict(float)
    ac_all_metrics = defaultdict(float)

    metrics , ac_metrics = model.test_step(model, answers, args, dataloader, query_name_dict)
    # metrics = model.test_step(model, answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    ac_num_queries = 0
    ac_num_query_structures = 0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]),
                              metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for query_structure in ac_metrics:
        log_metrics(mode + " ac " + query_name_dict[query_structure], step, ac_metrics[query_structure])
        for metric in ac_metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]),
                              ac_metrics[query_structure][metric], step)
            ac_all_metrics["_".join([query_name_dict[query_structure], metric])] = ac_metrics[query_structure][metric]
            if metric != 'num_queries':
                ac_average_metrics[metric] += ac_metrics[query_structure][metric]
        ac_num_queries += ac_metrics[query_structure]['num_queries']
        ac_num_query_structures += 1


    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)
    hit = -1

    for metric in ac_average_metrics:
        ac_average_metrics[metric] /= ac_num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), ac_average_metrics[metric], step)
        ac_all_metrics["_".join(["average", metric])] = ac_average_metrics[metric]
    log_metrics('%s average_ac' % mode, step, ac_average_metrics)

    return hit


def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_answers = pickle.load(open(os.path.join(args.data_path, "valid-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_answers = pickle.load(open(os.path.join(args.data_path, "test-answers.pkl"), 'rb'))

    for name in all_tasks:
        if name not in tasks:
            query_structure = name_query_dict[name]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, valid_queries, valid_answers, test_queries, test_answers


def main(args):
    max_hit = -1
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    args.do_train = False
    args.do_valid = False
    args.do_test = True
    args.print_on_screen = True
    cur_time = parse_time() + '_' + str(args.negative_sample_size) + '-' + args.data_path.split('/')[-1]
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-3] + '_' + args.data_path.split('/')[-2])

    tmp_str = "g-{}-d-{}-mode-{}".format(args.gamma, args.hidden_dim, args.beta_mode)

    if args.checkpoint_path is not None:

        args.save_path = args.checkpoint_path
    else:

        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    if not args.do_train:

        writer = SummaryWriter('./logs-debug/unused-tb')
    else:

        writer = SummaryWriter(args.save_path)

    set_logger(args)

    print("read entity and relation")
    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('-------------------------------' * 3)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    train_queries, train_answers, valid_queries, valid_answers, test_queries, test_answers = load_data(args, tasks)


    logging.info("Training info:")

    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))

        train_path_queries = flatten_query(train_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
                valid_answers
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                args.nentity,
                args.nrelation,
                test_answers
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        use_cuda=args.cuda,
        beta_mode=eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    args.checkpoint_path = "trained_model/PWA/data_PW/g-60.0-d-512-mode-[1600,2]/1-order/checkpoint20000"
    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        # checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name))
        checkpoint = torch.load("trained_model/PWA/data_PW/g-60.0-d-512-mode-[1600,2]/1-order/checkpoint20000")
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing Model...')
        init_step = 0

    step = init_step
    logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []
        for step in range(init_step, args.max_steps):


            log = model.train_step(model, optimizer, train_path_iterator, args)
            for metric in log:
                writer.add_scalar('path_' + metric, log[metric], step)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    evaluate(model, valid_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer)



            if step % args.save_checkpoint_steps == 0 and step > 0:
                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    evaluate(model, test_answers, args, test_dataloader, query_name_dict, 'Test', step,
                                       writer)

                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args, step)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args, step)
        print("the content of relation embedding: {}".format(model.relation_embedding))

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        evaluate(model, test_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)

    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())
