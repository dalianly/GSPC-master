import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import Counter
import pickle


# 定义图数据集类
class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Twitter15', 'Twitter15graph')):
        """
        初始化图数据集

        Args:
            fold_x (list): 数据集标识符列表
            treeDic (dict): 包含图数据的字典
            lower (int): 最小节点数
            upper (int): 最大节点数
            droprate (float): 边dropout比率
            data_path (str): 数据存储路径
        """
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        """返回数据集的大小"""
        return len(self.fold_x)

    def __getitem__(self, index):
        """
        获取单个样本

        Args:
            index (int): 样本索引

        Returns:
            Data: torch_geometric 数据对象
        """
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']

        # 如果设置了dropout率，则对边进行dropout操作
        if self.droprate > 0:
            row, col = list(edgeindex[0]), list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row, col = list(np.array(row)[poslist]), list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.LongTensor(data['root']),
            rootindex=torch.LongTensor([int(data['rootindex'])])
        )


# 定义双向图数据集类
class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Twitter16', 'Twitter16graph')):
        """
        初始化双向图数据集

        Args:
            fold_x (list): 数据集标识符列表
            treeDic (dict): 包含图数据的字典
            lower (int): 最小节点数
            upper (int): 最大节点数
            tddroprate (float): 时序边dropout比率
            budroprate (float): 用户-新闻边dropout比率
            data_path (str): 数据存储路径
        """
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        """返回数据集的大小"""
        return len(self.fold_x)

    def __getitem__(self, index):
        """
        获取单个样本

        Args:
            index (int): 样本索引

        Returns:
            Data: torch_geometric 数据对象
        """
        id = self.fold_x[index]
        uid = self.load_user_id(id)
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']

        # 时序边dropout操作
        if self.tddroprate > 0:
            row, col = list(edgeindex[0]), list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row, col = list(np.array(row)[poslist]), list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        # 用户-新闻边dropout操作
        burow, bucol = list(edgeindex[1]), list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            burow, bucol = list(np.array(burow)[poslist]), list(np.array(bucol)[poslist])
            bunew_edgeindex = [burow, bucol]
        else:
            bunew_edgeindex = [burow, bucol]

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            BU_edge_index=torch.LongTensor(bunew_edgeindex),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.LongTensor(data['root']),
            rootindex=torch.LongTensor([int(data['rootindex'])]),
            uid=torch.LongTensor(uid)
        )

    def load_user_id(self, id):
        """
        根据图标识符加载用户ID

        Args:
            id (str): 图的标识符

        Returns:
            list: 用户ID列表
        """
        userPath = os.path.join(cwd, 'data', 'Twitter16', 'Twitter16.txt')
        uid = []
        for line in open(userPath):
            line = line.rstrip()
            uid0, twitterID = line.split('\t')[0], line.split('\t')[1]
            if id == twitterID:
                uid.append(uid0)

        uid_counter = Counter(uid)
        X_uids = [K for K, V in uid_counter.most_common()]
        Uids = {idx: i + 1 for i, idx in enumerate(X_uids)}
        X_uid = [Uids[uid] for uid in uid]
        return X_uid


# 定义用户-新闻图数据集类
class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Twitter15', 'Twitter15graph')):
        """
        初始化用户-新闻图数据集

        Args:
            fold_x (list): 数据集标识符列表
            treeDic (dict): 包含图数据的字典
            lower (int): 最小节点数
            upper (int): 最大节点数
            droprate (float): 边dropout比率
            data_path (str): 数据存储路径
        """
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        """返回数据集的大小"""
        return len(self.fold_x)

    def __getitem__(self, index):
        """
        获取单个样本

        Args:
            index (int): 样本索引

        Returns:
            Data: torch_geometric 数据对象
        """
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row, col = list(edgeindex[0]), list(edgeindex[1])
        burow, bucol = list(edgeindex[1]), list(edgeindex[0])

        row.extend(burow)
        col.extend(bucol)

        # 如果设置了dropout率，则对边进行dropout操作
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row, col = list(np.array(row)[poslist]), list(np.array(col)[poslist])

        new_edgeindex = [row, col]

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.LongTensor(data['root']),
            rootindex=torch.LongTensor([int(data['rootindex'])
