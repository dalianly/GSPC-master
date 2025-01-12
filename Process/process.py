import os
from Process.dataset import GraphDataset, BiGraphDataset, UdGraphDataset
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import Counter


class TwitterTreeLoader:
    def __init__(self, dataname, cwd):
        self.dataname = dataname
        self.cwd = cwd
        self.tree_dic = {}

    def load_tree(self):
        if 'Twitter' in self.dataname:
            tree_path = os.path.join(self.cwd, 'data', self.dataname, 'data.TD_RvNN.vol_5000.txt')
            print("Reading Twitter tree...")
            for line in open(tree_path):
                line = line.rstrip()
                eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
                max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
                if eid not in self.tree_dic:
                    self.tree_dic[eid] = {}
                self.tree_dic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
            print('Tree count:', len(self.tree_dic))

        return self.tree_dic


class DatasetLoader:
    def __init__(self, dataname, cwd):
        self.dataname = dataname
        self.cwd = cwd
        self.tree_loader = TwitterTreeLoader(dataname, cwd)

    def load_data(self, fold_x_train, fold_x_test, droprate):
        tree_dic = self.tree_loader.load_tree()
        data_path = os.path.join(self.cwd, 'data', self.dataname + 'graph')

        print("Loading train set...")
        traindata_list = GraphDataset(fold_x_train, tree_dic, droprate=droprate, data_path=data_path)
        print("Train count:", len(traindata_list))

        print("Loading test set...")
        testdata_list = GraphDataset(fold_x_test, tree_dic, data_path=data_path)
        print("Test count:", len(testdata_list))

        return traindata_list, testdata_list

    def load_ud_data(self, fold_x_train, fold_x_test, droprate):
        tree_dic = self.tree_loader.load_tree()
        data_path = os.path.join(self.cwd, 'data', self.dataname + 'graph')

        print("Loading train set...")
        traindata_list = UdGraphDataset(fold_x_train, tree_dic, droprate=droprate, data_path=data_path)
        print("Train count:", len(traindata_list))

        print("Loading test set...")
        testdata_list = UdGraphDataset(fold_x_test, tree_dic, data_path=data_path)
        print("Test count:", len(testdata_list))

        return traindata_list, testdata_list

    def load_bi_data(self, fold_x_train, fold_x_test, TDdroprate, BUdroprate):
        tree_dic = self.tree_loader.load_tree()
        data_path = os.path.join(self.cwd, 'data', self.dataname, self.dataname + 'graph')

        print("Loading train set...")
        traindata_list = BiGraphDataset(fold_x_train, tree_dic, tddroprate=TDdroprate, budroprate=BUdroprate,
                                        data_path=data_path)
        print("Train count:", len(traindata_list))

        print("Loading test set...")
        testdata_list = BiGraphDataset(fold_x_test, tree_dic, data_path=data_path)
        print("Test count:", len(testdata_list))

        return traindata_list, testdata_list


class UserIdLoader:
    def __init__(self, dataname, cwd):
        self.dataname = dataname
        self.cwd = cwd

    def load_user_id(self, X_train, X_test):
        user_path = os.path.join(self.cwd, 'data', self.dataname, self.dataname + '.txt')
        train_uid = []
        test_uid = []

        for line in open(user_path):
            line = line.rstrip()
            uid0, twitterID = line.split('\t')[0], line.split('\t')[1]
            for i in X_train:
                if i == twitterID:
                    train_uid.append(uid0)
            for j in X_test:
                if j == twitterID:
                    test_uid.append(uid0)

        uid_counter = Counter(train_uid + test_uid)
        X_uids = [K for K, V in uid_counter.most_common()]
        Uids = {idx: i + 1 for i, idx in enumerate(X_uids)}
        X_train_uid = [Uids[uid] for uid in train_uid]
        X_test_uid = [Uids[uid] for uid in test_uid]

        return X_train_uid, X_test_uid


class CustomDataset(Dataset):
    def __init__(self, data_list, uid_list):
        self.data_list = data_list
        self.uid_list = uid_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        uid = self.uid_list[idx]
        return data, uid


# Example of how to use:
if __name__ == '__main__':
    # Define the dataset name and current working directory
    dataname = "Twitter"  # or "Weibo"
    cwd = os.getcwd()

    # Initialize DatasetLoader to load and process the data
    dataset_loader = DatasetLoader(dataname, cwd)

    # Example fold_x_train and fold_x_test
    fold_x_train = []  # Replace with actual training data
    fold_x_test = []  # Replace with actual test data
    droprate = 0.2  # Drop rate for the dataset (for training)

    # Load data
    traindata_list, testdata_list = dataset_loader.load_data(fold_x_train, fold_x_test, droprate)

    # Optionally, you can also load user IDs
    user_loader = UserIdLoader(dataname, cwd)
    X_train_uid, X_test_uid = user_loader.load_user_id(fold_x_train, fold_x_test)

    # Now you can wrap these data into CustomDataset for use in DataLoader
    train_dataset = CustomDataset(traindata_list, X_train_uid)
    test_dataset = CustomDataset(testdata_list, X_test_uid)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
