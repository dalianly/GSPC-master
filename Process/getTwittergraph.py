# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys


class NodeTweet:
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


class TwitterTreeProcessor:
    def __init__(self, obj, cwd):
        self.obj = obj
        self.cwd = cwd
        self.tree_dic = {}
        self.label_dic = {}
        self.tree_path = os.path.join(cwd, 'data', obj, 'data.TD_RvNN.vol_5000.txt')
        self.label_path = os.path.join(cwd, 'data', obj, f"{obj}_label_All.txt")

    def load_twitter_tree(self):
        print("Reading Twitter tree...")
        for line in open(self.tree_path):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

            if eid not in self.tree_dic:
                self.tree_dic[eid] = {}
            self.tree_dic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

        print(f'Tree nodes: {len(self.tree_dic)}')

    def load_labels(self):
        print("Loading tree labels...")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        event, y = [], []
        l1 = l2 = l3 = l4 = 0

        for line in open(self.label_path):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            label = label.lower()
            event.append(eid)
            if label in labelset_nonR:
                self.label_dic[eid] = 0
                l1 += 1
            if label in labelset_f:
                self.label_dic[eid] = 1
                l2 += 1
            if label in labelset_t:
                self.label_dic[eid] = 2
                l3 += 1
            if label in labelset_u:
                self.label_dic[eid] = 3
                l4 += 1
        print(f"Total labels: {len(self.label_dic)}")
        print(f"news: {l1}, false: {l2}, true: {l3}, unverified: {l4}")

    def str2matrix(self, Str):
        wordFreq, wordIndex = [], []
        for pair in Str.split(' '):
            freq = float(pair.split(':')[1])
            index = int(pair.split(':')[0])
            if index <= 5000:
                wordFreq.append(freq)
                wordIndex.append(index)
        return wordFreq, wordIndex

    def construct_mat(self, tree):
        index2node = {}
        for i in tree:
            node = NodeTweet(idx=i)
            index2node[i] = node
        for j in tree:
            indexC = j
            indexP = tree[j]['parent']
            nodeC = index2node[indexC]
            wordFreq, wordIndex = self.str2matrix(tree[j]['vec'])
            nodeC.index = wordIndex
            nodeC.word = wordFreq
            if indexP != 'None':
                nodeP = index2node[int(indexP)]
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
            else:
                rootindex = indexC - 1
                root_index = nodeC.index
                root_word = nodeC.word
        rootfeat = np.zeros([1, 5000])
        if len(root_index) > 0:
            rootfeat[0, np.array(root_index)] = np.array(root_word)
        matrix = np.zeros([len(index2node), len(index2node)])
        row, col, x_word, x_index = [], [], [], []
        for index_i in range(len(index2node)):
            for index_j in range(len(index2node)):
                if index2node[index_i + 1].children and index2node[index_j + 1] in index2node[index_i + 1].children:
                    matrix[index_i][index_j] = 1
                    row.append(index_i)
                    col.append(index_j)
            x_word.append(index2node[index_i + 1].word)
            x_index.append(index2node[index_i + 1].index)
        edgematrix = [row, col]
        return x_word, x_index, edgematrix, rootfeat, rootindex

    def get_feature(self, x_word, x_index):
        x = np.zeros([len(x_index), 5000])
        for i in range(len(x_index)):
            if len(x_index[i]) > 0:
                x[i, np.array(x_index[i])] = np.array(x_word[i])
        return x


class DatasetLoader:
    def __init__(self, obj, cwd, tree_processor):
        self.obj = obj
        self.cwd = cwd
        self.tree_processor = tree_processor
        self.event = []

    def load_data(self):
        self.tree_processor.load_twitter_tree()
        self.tree_processor.load_labels()

        print("Loading dataset...")
        Parallel(n_jobs=30, backend='threading')(
            delayed(self.load_eid)(eid) for eid in tqdm(self.tree_processor.label_dic)
        )

    def load_eid(self, eid):
        if eid not in self.tree_processor.tree_dic:
            return None
        x_word, x_index, tree, rootfeat, rootindex = self.tree_processor.construct_mat(
            self.tree_processor.tree_dic[eid])
        x_x = self.tree_processor.get_feature(x_word, x_index)
        rootfeat, tree, x_x, rootindex = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(rootindex)
        y = np.array([self.tree_processor.label_dic[eid]])
        np.savez(os.path.join(self.cwd, 'data', self.obj, 'graph', f'{eid}.npz'), x=x_x, root=rootfeat, edgeindex=tree,
                 rootindex=rootindex, y=y)


def main(obj):
    cwd = os.getcwd()
    tree_processor = TwitterTreeProcessor(obj, cwd)
    dataset_loader = DatasetLoader(obj, cwd, tree_processor)
    dataset_loader.load_data()


if __name__ == '__main__':
    obj = sys.argv[1]
    main(obj)
