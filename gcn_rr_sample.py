#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import torch
import scipy.sparse as sp
import math
from torch import nn
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# In[27]:


def load_data_edges(path="cora.tgz_files/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype = np.int32)
    idx_map = {j : i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path,dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten())), dtype = np.int32).reshape(edges_unordered.shape)
    idx_train = range(200)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return edges, features, labels, idx_train, idx_val, idx_test


# In[186]:


def p_value(eps, n=2):
    return np.e ** eps / (np.e ** eps + n - 1)
def random_response(data,eps_first):
    for user in data:
        if user.sum() <= 0:
            pass
        else:
            eps = eps_first
            eps = eps / user.sum()
            #print("eps is {} ,user.sum is {}".format(eps, user.sum()))
            p = p_value(eps)
            #print("p_value is {}".format(round(p,10)))
            for i in range(len(user)):
                if user[i] == 1:
                    user[i] = np.random.binomial(1, p)


# In[187]:


from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class GraphConvolution(Module):
        def __init__(self, in_features, out_features, bias=True):
            super(GraphConvolution, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else :
                self.register_parameter('bias', None)
            self.reset_parameter()
        def reset_parameter(self):
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        def forward(self, input, adj):
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                output = output + self.bias
                return output
            else:
                return output
        def __repr__(self):
            return self.__class__.__name__ + '(' + str(self.in_features) + '->'+ str(self.out_features) + ')'  
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# In[189]:


import sys
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
edges, features, labels, idx_train, idx_val, idx_test = load_data_edges()
model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.8)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)
def train(epoch):
    matrix = np.zeros((2708,2708))
    for i in range(5429):
        matrix[edges[i, 0]][edges[i, 1]] = 1
    t = time.time()
    model.train()
    optimizer.zero_grad()
    print(len(sp.csr_matrix(sp.coo_matrix(matrix)).data))
    random_response(matrix, 0.02)
    print(len(sp.csr_matrix(sp.coo_matrix(matrix)).data))
    adj = sp.coo_matrix(matrix)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 计算准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 反向求导  Back Propagation
    loss_train.backward()
    # 更新所有的参数
    optimizer.step()
    # 验证集的损失函数
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}'.format(time.time() - t))
def test():
    model.eval()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0],edges[:,1])), shape=(labels.shape[0],labels.shape[0]),dtype = np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
t_total = time.time()
for epoch in range(100):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()
torch.cuda.empty_cache()


# In[178]:


matrix = np.zeros((2708,2708))
for i in range(5429):
    matrix[edges[i, 0]][edges[i, 1]] = 1
matrix.sum()


# In[184]:


random_response(matrix, 100)
matrix.sum()


# In[125]:


def p_value(eps, n=2):
    return np.e ** eps / (np.e ** eps + n - 1)
def random_response(data,eps):
    for user in data:
        if user.sum() <= 0:
            pass
        else:
            eps = eps / user.sum()
            p = p_value(eps)
            for i in range(len(user)):
                if user[i] == 1:
                    user[i] = np.random.binomial(1, p)


# In[164]:


p_value(1000/19)


# In[ ]:




