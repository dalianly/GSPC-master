import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import sys

class FakeNewsAttention(nn.Module):
    def __init__(self, config):
        super(FakeNewsAttention, self).__init__()
        self.config = config
        self.n_heads = config['n_heads']
        self.A_us = config['A_us']
        self.device = 'cuda:0'
        embeding_size = config['embeding_size']

        # 初始化用户嵌入层
        self.user_embedding = nn.Embedding(2213, embeding_size, padding_idx=0)

        # 卷积层配置
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=50 - K + 1) for K in config['kernel_sizes']])

        # 初始化多头注意力的权重
        self.Wcm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in range(self.n_heads)]
        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).cuda()

        # 用户输出层的权重
        self.W1 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W2 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.linear = nn.Linear(400, 200)

        # 激活函数和正则化
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        # 用户输出层
        self.fc_user_out = nn.Sequential(
            nn.Linear(embeding_size, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 128)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        权重初始化方法，使用 Xavier 初始化方法。
        """
        init.xavier_normal_(self.user_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.linear.weight)
        for name, param in self.fc_user_out.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)

    def forward(self, X_user_id):
        """
        模型的前向传播过程。

        Args:
            X_user_id: 输入的用户ID (LongTensor)。

        Returns:
            user_rep: 计算出的用户表示。
        """
        X_user_id = torch.LongTensor(X_user_id).to(self.device)
        X_user = self.user_embedding(X_user_id)

        m_hat = []
        for i in range(self.n_heads):
            M = self.source_embedding.weight
            linear1 = torch.einsum("bd,dd,sd->bs", X_user, self.Wcm[i], M) / self.scale
            linear1 = self.relu(linear1)

            # 获取用户注意力矩阵并应用 softmax
            A_us = self.A_us[X_user_id.cpu(), :].todense()
            A_us = torch.FloatTensor(A_us).cuda()

            alpha = F.softmax(linear1 * A_us, dim=-1)
            alpha = self.dropout(alpha)
            alpha = alpha.matmul(M)
            m_hat.append(alpha)

        # 拼接并处理多头注意力结果
        m_hat = torch.cat(m_hat, dim=-1).matmul(self.W1)
        m_hat = self.elu(m_hat)
        m_hat = self.dropout(m_hat)

        # 计算最终的用户表示
        user_rep = m_hat + X_user
        Xu_logit = self.fc_user_out(user_rep)
        return user_rep
