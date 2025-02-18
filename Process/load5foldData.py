import random
from random import shuffle
import os


def load5foldData(obj):
    # Twitter数据集
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd, "data", obj, obj + "_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("Loading tree label...")

        NR, F, T, U = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        labelDic = {}

        for line in open(labelPath):
            line = line.rstrip()
            label, eid, t = line.split('\t')[0], line.split('\t')[2], line.split('\t')[8]
            t = float(t)
            labelDic[eid] = label.lower()

            if t < 1:
                if label in labelset_nonR:
                    NR.append(eid)
                    l1 += 1
                if labelDic[eid] in labelset_f:
                    F.append(eid)
                    l2 += 1
                if labelDic[eid] in labelset_t:
                    T.append(eid)
                    l3 += 1
                if labelDic[eid] in labelset_u:
                    U.append(eid)
                    l4 += 1

        print(len(labelDic))
        print(l1, l2, l3, l4)

        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)

        # Initialize lists for fold splits
        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []

        # Calculate fold sizes
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        # Split into 5 folds
        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])

        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1 * 2])
        fold1_x_test.extend(F[leng2:leng2 * 2])
        fold1_x_test.extend(T[leng3:leng3 * 2])
        fold1_x_test.extend(U[leng4:leng4 * 2])

        fold2_x_train.extend(NR[0:leng1 * 2])
        fold2_x_train.extend(NR[leng1 * 3:])
        fold2_x_train.extend(F[0:leng2 * 2])
        fold2_x_train.extend(F[leng2 * 3:])
        fold2_x_train.extend(T[0:leng3 * 2])
        fold2_x_train.extend(T[leng3 * 3:])
        fold2_x_train.extend(U[0:leng4 * 2])
        fold2_x_train.extend(U[leng4 * 3:])
        fold2_x_test.extend(NR[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(F[leng2 * 2:leng2 * 3])
        fold2_x_test.extend(T[leng3 * 2:leng3 * 3])
        fold2_x_test.extend(U[leng4 * 2:leng4 * 3])

        fold3_x_train.extend(NR[0:leng1 * 3])
        fold3_x_train.extend(NR[leng1 * 4:])
        fold3_x_train.extend(F[0:leng2 * 3])
        fold3_x_train.extend(F[leng2 * 4:])
        fold3_x_train.extend(T[0:leng3 * 3])
        fold3_x_train.extend(T[leng3 * 4:])
        fold3_x_train.extend(U[0:leng4 * 3])
        fold3_x_train.extend(U[leng4 * 4:])
        fold3_x_test.extend(NR[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(F[leng2 * 3:leng2 * 4])
        fold3_x_test.extend(T[leng3 * 3:leng3 * 4])
        fold3_x_test.extend(U[leng4 * 3:leng4 * 4])

        fold4_x_train.extend(NR[0:leng1 * 4])
        fold4_x_train.extend(NR[leng1 * 5:])
        fold4_x_train.extend(F[0:leng2 * 4])
        fold4_x_train.extend(F[leng2 * 5:])
        fold4_x_train.extend(T[0:leng3 * 4])
        fold4_x_train.extend(T[leng3 * 5:])
        fold4_x_train.extend(U[0:leng4 * 4])
        fold4_x_train.extend(U[leng4 * 5:])
        fold4_x_test.extend(NR[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(F[leng2 * 4:leng2 * 5])
        fold4_x_test.extend(T[leng3 * 4:leng3 * 5])
        fold4_x_test.extend(U[leng4 * 4:leng4 * 5])

    # Weibo 数据集处理
    elif obj == "Weibo":
        labelPath = os.path.join(cwd, "data", "Weibo", "weibo_id_label.txt")
        print("Loading Weibo label...")

        F, T = [], []
        l1 = l2 = 0
        labelDic = {}

        for line in open(labelPath):
            line = line.rstrip()
            eid, label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)

            if labelDic[eid] == 0:
                F.append(eid)
                l1 += 1
            if labelDic[eid] == 1:
                T.append(eid)
                l2 += 1

        print(len(labelDic))
        print(l1, l2)

        random.shuffle(F)
        random.shuffle(T)

        # Initialize lists for fold splits
        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []

        # Calculate fold sizes
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)

        # Split into 5 folds
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold
