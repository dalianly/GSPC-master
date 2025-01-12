import sys
import os
import pickle
import torch as th
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from tools.earlystopping import EarlyStopping
from tools.evaluate import *
from tqdm import tqdm
from Process.process import *
from Process.load5foldData import *
import numpy as np

class FakeNewsGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(FakeNewsGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = x.float().clone()
        x = self.conv1(x, edge_index)
        x2 = x.clone()

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x

# 使用用户-新闻图数据和用户兴趣图数据的 GCN 模型
class UserNewsGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(UserNewsGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = x.float().clone()
        x = self.conv1(x, edge_index)
        x2 = x.clone()

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x

# 基于图卷积网络的多头注意力机制模型，用于虚假新闻检测
class FakeNewsAttention(th.nn.Module):

    def __init__(self, config):
        super(FakeNewsAttention, self).__init__()
        self.config = config
        self.n_heads = config['n_heads']
        self.A_us = config['A_us']
        self.device = 'cuda:0'
        embeding_size = batchsize  # 需要确认

        self.user_embedding = th.nn.Embedding(config['A_us'].shape[0], embeding_size, padding_idx=0)
        self.source_embedding = th.nn.Embedding(config['A_us'].shape[1], embeding_size)

        self.convs = th.nn.ModuleList([th.nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = th.nn.ModuleList([th.nn.MaxPool1d(kernel_size=50 - K + 1) for K in config['kernel_sizes']])

        self.Wcm = [th.nn.Parameter(th.FloatTensor(embeding_size, embeding_size)).cuda() for _ in range(self.n_heads)]
        self.Wam = [th.nn.Parameter(th.FloatTensor(embeding_size, embeding_size)).cuda() for _ in range(self.n_heads)]
        self.scale = th.sqrt(th.FloatTensor([embeding_size])).cuda()

        self.W1 = th.nn.Parameter(th.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W2 = th.nn.Parameter(th.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.linear = th.nn.Linear(400, 200)

        self.dropout = th.nn.Dropout(config['dropout'])
        self.relu = th.nn.ReLU()
        self.elu = th.nn.ELU()

        self.fc_user_out = th.nn.Sequential(
            th.nn.Linear(embeding_size, 128),
            th.nn.ReLU(),
            th.nn.Dropout(config['dropout']),
            th.nn.Linear(128, 128)
        )

        init.xavier_normal_(self.user_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.linear.weight)
        for name, param in self.fc_user_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def forward(self, data):
        X_user_id = data.uid
        X_user = self.user_embedding(X_user_id)
        m_hat = []

        for i in range(self.n_heads):
            M = self.source_embedding.weight
            linear1 = th.einsum("bd,dd,sd->bs", X_user, self.Wcm[i], M) / self.scale
            linear1 = self.relu(linear1)

            A_us = self.A_us[X_user_id.cpu(), :].todense()
            A_us = th.FloatTensor(A_us).cuda()

            alpha = F.softmax(linear1 * A_us, dim=-1)
            alpha = self.dropout(alpha)
            alpha = alpha.matmul(M)
            m_hat.append(alpha)

        m_hat = th.cat(m_hat, dim=-1).matmul(self.W1)
        m_hat = self.elu(m_hat)
        m_hat = self.dropout(m_hat)

        user_rep = m_hat + X_user
        Xu_logit = self.fc_user_out(user_rep)
        return user_rep


# 综合虚假新闻检测模型，结合了图卷积和多头注意力机制
class FakeNewsModel(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, config):
        super(FakeNewsModel, self).__init__()
        self.FakeNewsAttention = FakeNewsAttention(config)
        self.FakeNewsGCN = FakeNewsGCN(in_feats, hid_feats, out_feats)
        self.UserNewsGCN = UserNewsGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 3, 4)  # 预测类别数量为4

    def forward(self, data):
        FN_x = self.FakeNewsGCN(data)
        UN_x = self.UserNewsGCN(data)
        User_x = self.FakeNewsAttention(data)
        x = th.cat((UN_x, FN_x, User_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# 训练函数，适用于虚假新闻检测任务
def train_fake_news_model(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter):
    # 载入数据
    file = os.path.join(cwd, 'data/' + dataname + '/relations.pkl')
    with open(file, "rb+") as fp:
        A_us, A_uu = pickle.load(fp, encoding="latin1")

    config['A_us'] = A_us
    model = FakeNewsModel(5000, 64, 64, config).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    train_losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(treeDic)
        loss = F.nll_loss(out, x_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return model, train_losses
