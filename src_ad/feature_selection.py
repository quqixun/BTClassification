from __future__ import print_function


import os
import json
import numpy as np
from tqdm import *
import pandas as pd
from random import shuffle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def load_data(info, mode):
    x, y = [], []
    print("Loading {} data ...".format(mode))
    for subject in info:
        dir_path = subject[0]
        label = subject[1]

        y.append(label)
        x.append(np.load(os.path.join(dir_path, "fc1.npy")).ravel())

    x = np.array(x)
    y = np.array(y)

    return x, y


def load_paras(file_path, model):
    paras = json.load(open(file_path))
    return paras[model]


def get_sepdata_path(data_dir):
    groups = os.listdir(data_dir)
    data_info = []
    for group in groups:
        if group == "AD":
            label = 1
        else:
            label = 0

        group_dir = os.path.join(data_dir, group)
        for subj in os.listdir(group_dir):
            subj_dir = os.path.join(group_dir, subj)
            data_info.append([subj_dir, label])

    shuffle(data_info)
    return data_info


def reduce_dimension(x, y=None, n_components=100, method="pca"):
    if method == "pca":
        rd = PCA(n_components=n_components)
        rd.fit(x)
    elif method == "lda":
        rd = LDA()
        rd.fit(x, y)
    return rd


def classifier(train_x, train_y,
               valid_x, valid_y,
               test_x, test_y):
    model1 = RFC(n_estimators=200,
                 max_depth=3,
                 criterion="entropy",
                 # max_features="log2",
                 # min_samples_split=5,
                 # min_samples_leaf=5,
                 # random_state=9001,
                 n_jobs=10)

    model1.fit(train_x, train_y)
    train_pred = model1.predict(train_x)
    valid_pred = model1.predict(valid_x)
    test_pred = model1.predict(test_x)

    print("RFC Results:")

    train_acc = accuracy_score(train_y, train_pred)
    print("Train Acc:", train_acc)

    valid_acc = accuracy_score(valid_y, valid_pred)
    print("Valid Acc", valid_acc)

    test_acc = accuracy_score(test_y, test_pred)
    print("Test Acc:", test_acc)

    importance = model1.feature_importances_
    df = pd.DataFrame(data={"Importance": importance})
    df.to_csv("importances.csv", index=False)

    idx = np.where(importance >= 5e-4)[0]
    print("Important features:", len(idx))
    imp_train_x = train_x[:, idx]
    imp_valid_x = valid_x[:, idx]
    imp_test_x = test_x[:, idx]

    # model2 = MLPClassifier(hidden_layer_sizes=(256, ),
    #                        batch_size=64,
    #                        max_iter=1000, alpha=5e-5,
    #                        solver='adam', verbose=0,
    #                        tol=1e-6,
    #                        activation="tanh",
    #                        # random_state=99001,
    #                        learning_rate_init=1e-4)

    model2 = SVC(C=1, kernel="rbf",
                 degree=4, gamma="auto",
                 verbose=True, tol=1e-4)

    model2.fit(imp_train_x, train_y)
    train_pred = model2.predict(imp_train_x)
    valid_pred = model2.predict(imp_valid_x)
    test_pred = model2.predict(imp_test_x)

    print("NN Results:")

    train_acc = accuracy_score(train_y, train_pred)
    print("Train Acc:", train_acc)

    valid_acc = accuracy_score(valid_y, valid_pred)
    print("Valid Acc", valid_acc)

    test_acc = accuracy_score(test_y, test_pred)
    print("Test Acc:", test_acc)

    return valid_acc, test_acc


def lda(train_x, train_y,
        valid_x, valid_y,
        test_x, test_y):

    model = LDA()
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)
    test_pred = model.predict(test_x)

    print("NN Results:")

    train_acc = accuracy_score(train_y, train_pred)
    print("Train Acc:", train_acc)

    valid_acc = accuracy_score(valid_y, valid_pred)
    print("Valid Acc", valid_acc)

    test_acc = accuracy_score(test_y, test_pred)
    print("Test Acc:", test_acc)

    return valid_acc, test_acc


def adaboost(train_x, train_y,
             valid_x, valid_y,
             test_x, test_y):

    clf = DTC(criterion="gini",
              max_depth=5,
              random_state=325)

    model = ABC(base_estimator=clf,
                n_estimators=200,
                # random_state=727
                )

    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)
    test_pred = model.predict(test_x)

    print("NN Results:")

    train_acc = accuracy_score(train_y, train_pred)
    print("Train Acc:", train_acc)

    valid_acc = accuracy_score(valid_y, valid_pred)
    print("Valid Acc", valid_acc)

    test_acc = accuracy_score(test_y, test_pred)
    print("Test Acc:", test_acc)

    return valid_acc, test_acc


def svm(train_x, train_y,
        valid_x, valid_y,
        test_x, test_y):
    model = SVC(C=1,
                kernel="rbf",
                degree=2,
                gamma=0.0001,
                verbose=True,
                tol=1e-4)

    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)
    test_pred = model.predict(test_x)

    print("NN Results:")

    train_acc = accuracy_score(train_y, train_pred)
    print("Train Acc:", train_acc)

    valid_acc = accuracy_score(valid_y, valid_pred)
    print("Valid Acc", valid_acc)

    test_acc = accuracy_score(test_y, test_pred)
    print("Test Acc:", test_acc)

    return valid_acc, test_acc


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "features")
    trainset_info = get_sepdata_path(os.path.join(data_dir, "train"))
    validset_info = get_sepdata_path(os.path.join(data_dir, "valid"))
    testset_info = get_sepdata_path(os.path.join(data_dir, "test"))

    # print(len(trainset_info), len(validset_info), len(testset_info))

    train_x, train_y = load_data(trainset_info, "trainset")
    valid_x, valid_y = load_data(validset_info, "validset")
    test_x, test_y = load_data(testset_info, "testset")

    valid_res, test_res = [], []
    # for i in tqdm(range(10)):
    #     # clf = lda
    #     # clf = classifier
    #     # clf = adaboost
    #     vres, tres = clf(train_x, train_y,
    #                      valid_x, valid_y,
    #                      test_x, test_y)
    #     valid_res.append(vres)
    #     test_res.append(tres)

    clf = svm
    vres, tres = clf(train_x, train_y,
                     valid_x, valid_y,
                     test_x, test_y)
    valid_res.append(vres)
    test_res.append(tres)

    print("Valid ACC: {0:.4f} +/- {1:.4f}".format(np.mean(valid_res), np.std(valid_res)))
    print("Test ACC: {0:.4f} +/- {1:.4f}".format(np.mean(test_res), np.std(test_res)))

    test_res = list(test_res)
    acc_set = list(set(test_res))
    acc_set.sort()
    print("Occurrence")
    for acc in acc_set:
        print(acc, "-", test_res.count(acc))
