#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from metrics import ndcg_at_k, MRR


class BetaIntersection(nn.Module):
    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)

        layer1_act = F.relu(self.layer1(all_embeddings))
        attention = F.softmax(self.layer2(layer1_act), dim=0)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)

        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):

        x = torch.cat([e_embedding, r_embedding], dim=-1)

        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

# 正则化处理
class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 test_batch_size=1, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2))
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)


        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        hidden_dim, num_layers = beta_mode
        self.center_net = BetaIntersection(self.entity_dim)
        self.projection_net = BetaProjection(self.entity_dim * 2,
                                             self.relation_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)


    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict , forward_type = "train" , para = None):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        # sample_alpha_embeddings , sample_beta_embeddings = [] , []
        for query_structure in batch_queries_dict:
            alpha_embedding, beta_embedding, _ = self.embed_query(batch_queries_dict[query_structure],
                                                                        query_structure,
                                                                        0 , forward_type)
            if batch_idxs_dict != None:
                all_idxs.extend(batch_idxs_dict[query_structure])

            all_alpha_embeddings.append(alpha_embedding)
            all_beta_embeddings.append(beta_embedding)


        # if len(sample_api) > 0:
        #     query_dict = collections.defaultdict(list)
        #     if len(sample_api) == 1:
        #         value = [sample_api[0], 0, sample_api[0], 1, -2]
        #         query_dict[(('e', ('r',)), ('e', ('r', 'n')))].append(value)
        #     else:
        #         value = [sample_api[0], 0, sample_api[0], 1, -2 , sample_api[1] , 0 , sample_api[1] , 1 , -2]
        #         query_dict[((('e', ('r',)), ('e', ('r', 'n'))), (('e', ('r',)), ('e', ('r', 'n'))))].append(value)
        #     for query_structure in query_dict:
        #         query_dict[query_structure] = torch.LongTensor(query_dict[query_structure]).cuda()
        #     for query_structure in query_dict:
        #         alpha_embedding , beta_embedding  , _ = self.embed_query(query_dict[query_structure] , query_structure , 0 , forward_type)
        #         sample_alpha_embeddings.append(alpha_embedding)
        #         sample_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:

            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)

            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)

            if para != None:
                para = para.unsqueeze(1).unsqueeze(1)
                all_alpha_embeddings = (all_alpha_embeddings * para).sum(dim = 0 , keepdim = True)
                all_beta_embeddings = (all_beta_embeddings * para).sum(dim = 0 , keepdim = True)

                # if len(sample_api) > 0:
                #     sample_alpha_embeddings = torch.cat(sample_alpha_embeddings , dim = 0).unsqueeze(1)
                #     sample_beta_embeddings = torch.cat(sample_beta_embeddings, dim = 0).unsqueeze(1)
                #     all_alpha_embeddings = torch.cat((all_alpha_embeddings , sample_alpha_embeddings) , dim = 0)
                #     all_beta_embeddings = torch.cat((all_beta_embeddings , sample_beta_embeddings) , dim = 0)

            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)



        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                if len(all_idxs) != 0:
                    positive_sample_regular = positive_sample[all_idxs]
                else:
                    positive_sample_regular = positive_sample
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))

                negative_logit = self.cal_logit(negative_embedding, all_dists)

            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs


    def embed_query(self, queries, query_structure, idx , forward_type):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        # print(queries)
        # print(query_structure)
        # print(query_structure[-1])
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n','h']:
                all_relation_flag = False
                break



        if all_relation_flag:

            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[0], idx , forward_type)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            # print(query_structure[-1])
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                elif query_structure[-1][i] == 'h':
                    assert (queries[:, idx] == -3).all()
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1

            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        else:
            # print("False")
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):

                alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[i], idx , forward_type)

                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))


        return alpha_embedding, beta_embedding, idx

    # 计算KL散度
    def cal_logit(self, entity_embedding, query_dist):

        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit



    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)

        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        # print("batch_query : " , batch_queries_dict)
        # print("batch_id : " , batch_idxs_dict)

        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict , forward_type = "train")

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log


    @staticmethod
    def test_step(model, answers, args, test_dataloader, query_name_dict, save_result=False, save_str="",
                  save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)
        dict = {}
        # 存放推荐结果
        rank_list_all = []
        # 存放真实结果
        hard_answer_all = []
        # 存放查询集
        li_all = []
        # 存放分数
        score_all = []
        with torch.no_grad():
            for positive_sample, negative_sample, subsampling_weight, queries, queries_unflatten, query_structures in tqdm(
                    test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                # print("queries : " , queries)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)


                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()
                    positive_sample = positive_sample.cuda()
                    subsampling_weight = subsampling_weight.cuda()
                positive_logit, negative_logit, subsampling_weight, idxs = model(positive_sample, negative_sample,
                                                                                 subsampling_weight, batch_queries_dict,
                                                                                 batch_idxs_dict , forward_type = "test")

                negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
                negative_sample_loss = - (subsampling_weight * negative_score).sum()

                negative_sample_loss /= subsampling_weight.sum()

                loss = negative_sample_loss
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                # 生成排序结果
                ranking_list = argsort[0].tolist()
                score = negative_logit[0].tolist()
                score = sorted(score , reverse = True)

                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = answers[query]
                    num_hard = len(hard_answer)
                    dict[query] = ranking_list

                    # if len(query) == 2:
                    #     if isinstance(query[0][0], int):
                    #         li = [query[0][0]]
                    #     else:
                    #         li = [query[0][0][0], query[1][0][0]]
                    # elif len(query) == 3:
                    #     li = [query[0][0][0], query[1][0][0], query[2][0][0]]

                    if len(query) == 2:
                        if isinstance(query[0], int):
                            li = [query[0]]
                        else:
                            li = [query[0][0], query[1][0]]
                    elif len(query) == 3:
                        li = [query[0][0], query[1][0], query[2][0]]

                    li_all.append(li)

                    # 去掉查询集中的云API后的排序结果
                    new_ranking_list = []
                    # 真实结果
                    new_hard_answer = []
                    # 对应的分数
                    new_score = []

                    for v in ranking_list:
                        if v not in li:
                            new_ranking_list.append(v)

                    for v in hard_answer:
                        if v not in li:
                            new_hard_answer.append(v)

                    for i , v in enumerate(ranking_list):
                        if v not in li:
                            new_score.append(score[i])
                    rank_list_all.append(new_ranking_list)
                    hard_answer_all.append(new_hard_answer)
                    score_all.append(new_score)
                    ndcg20 = ndcg_at_k(new_hard_answer, new_ranking_list, 20)
                    mrr20 = MRR(new_hard_answer, new_ranking_list, 20)


                    if query_structure not in logs:
                        logs[query_structure].append({
                            'NDCG@20': ndcg20,
                            'MRR@20': mrr20,
                            'loss': loss,
                            'num_queries': 1,
                            'num_hard_answer': num_hard,
                        })
                    else:
                        logs[query_structure][0]['NDCG@20'] += ndcg20
                        logs[query_structure][0]['MRR@20'] += mrr20
                        logs[query_structure][0]['loss'] += loss
                        logs[query_structure][0]['num_queries'] += 1

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1

        # print(len(score_all) , len(rank_list_all) , len(hard_answer_all) , len(li_all))

        # 分别对所有的一阶、二阶、三阶进行指标计算
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            metrics[query_structure]['num_queries'] = logs[query_structure][0]['num_queries']
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer', 'num_queries']:
                    continue
                metrics[query_structure][metric] = logs[query_structure][0][metric] / logs[query_structure][0][
                    'num_queries']

        print("**************进行溯因矫正*****************")
        # 1-order top_K = 80  weight = 0.15  num = 100
        # 2-order top_K = 30  weight = 0.2   num = 100
        # 3-order top_K = 50  weight = 0.7   num = 100
        # 123-order top_K = 60  weight = 0.7 num = 100
        # 2-order
        top_K = 80
        weight = 0.15
        num = 100
        ac_metrics = KGReasoning.abductive_correction(model , rank_list_all , hard_answer_all , li_all , score_all , top_K , weight , num , args.seed)
        return metrics , ac_metrics

    @staticmethod
    def get_parameter(pre_var , weight):
        para = torch.sigmoid(pre_var)
        para = (para ** weight) / (para ** weight).sum(-1)
        return para


    @staticmethod
    def abductive_correction(model , rank_list_all , hard_answer_all , li_all , score_all , top_K , weight , num , seed):
        model.eval()

        logs = collections.defaultdict(list)

        random.seed(seed)
        # 选取出一个真实原因
        real_reason_all = []
        for i in range(len(rank_list_all)):
            real_reason_all.append([])
        for i in range(len(li_all)):
            real_reason_all[i].append(random.choice(li_all[i]))



        for i in range(len(rank_list_all)):
            print(f"Num : [{i + 1} / {len(rank_list_all)}]")
            real_reason = torch.tensor(real_reason_all[i]).cuda()
            rank_list = rank_list_all[i][ : top_K]
            hard_answer = hard_answer_all[i]
            li = li_all[i]
            score = sorted(score_all[i][ : top_K] , reverse = True)
            score = torch.tensor(score).cuda()
            pre_var = Variable(torch.empty_like(score , requires_grad = True)).cuda()
            with torch.no_grad():
                pre_var.copy_(torch.from_numpy(score.cpu().numpy()).data)
            optimizer = optim.Adam([pre_var] , lr = 0.001)
            pre_var.requires_grad = True
            # 综合云API所包含的候选云API
            query_dict = collections.defaultdict(list)
            for i in range(len(rank_list)):
                value = [rank_list[i], 1, 0 , rank_list[i], 1, -2]
                query_dict[(('e', ('r', 'r')), ('e', ('r', 'n')))].append(value)
            for query_structrue in query_dict:
                query_dict[query_structrue] = torch.LongTensor(query_dict[query_structrue]).cuda()

            for n_ in range(num):
                para = KGReasoning.get_parameter(pre_var , weight)
                logit , _ , _  , _ = model(real_reason , None , None , query_dict , None , para = para)
                difference = torch.sigmoid(logit)
                eplison = 1e-8
                loss = -torch.log(torch.clamp(difference , min = eplison))
                loss = loss.unsqueeze(0)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([pre_var] , 0.5)
                optimizer.step()
            score = pre_var.tolist()
            sorted_pair = sorted(zip(score , rank_list) , reverse = True)
            score , rank_list = zip(*sorted_pair)
            if len(li) == 1:
                query_structrue = (('e', ('r', 'r')), ('e', ('r', 'n')))
            elif len(li) == 2:
                query_structrue = ((('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n'))))
            else:
                query_structrue = ((('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n'))), (('e', ('r', 'r')), ('e', ('r', 'n'))))


            ndcg20 = ndcg_at_k(hard_answer , rank_list , 20)
            mrr20 = MRR(hard_answer , rank_list , 20)
            num_hard = len(hard_answer)
            if query_structrue not in logs:
                logs[query_structrue].append({
                    "NDCG@20" : ndcg20 ,
                    "MRR" : mrr20 ,
                    "num_queries" : 1 ,
                    "num_hard_answer" : num_hard
                })
            else:
                logs[query_structrue][0]["NDCG@20"] += ndcg20
                logs[query_structrue][0]["MRR"] += mrr20
                logs[query_structrue][0]["num_queries"] += 1

            metrics = collections.defaultdict(lambda: collections.defaultdict(int))
            for query_structure in logs:
                metrics[query_structure]['num_queries'] = logs[query_structure][0]['num_queries']
                for metric in logs[query_structure][0].keys():
                    if metric in ['num_hard_answer', 'num_queries']:
                        continue
                    metrics[query_structure][metric] = logs[query_structure][0][metric] / logs[query_structure][0][
                        'num_queries']

        return metrics