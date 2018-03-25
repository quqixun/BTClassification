from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#
# Directories
#

parent_dir = os.path.dirname(os.getcwd())

# Load all results
res_csv = "valid_res.csv"
res_df = pd.read_csv(res_csv)


#
# Learning Curve
#

curve_csv = os.path.join(parent_dir, "models", "model0", "learning_curve.csv")
curve_df = pd.read_csv(curve_csv)

train_loss = curve_df["loss"].values.tolist()
train_acc = curve_df["acc"].values.tolist()
valid_loss = curve_df["val_loss"].values.tolist()
valid_acc = curve_df["val_acc"].values.tolist()

epoch_num = len(train_loss)
x = np.arange(1, epoch_num + 1)


plt.figure(num="loss_curve", figsize=(9, 5))
plt.plot(x, train_loss, label="Training Loss")
plt.plot(x, valid_loss, label="Validation Loss")
plt.plot([60, 60], [0, 1.3], "k--", lw=1, alpha=0.7)
plt.ylim([0, 1.3])
plt.xticks((0, 50, 60, 100, 150, 200, 250), (0, 50, 60, 100, 150, 200, 250), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid("on", linestyle="--", linewidth=0.5, alpha=0.5)
plt.show()


plt.figure(num="acc_curve", figsize=(9, 5))
plt.plot(x, train_acc, label="Training Accuracy")
plt.plot(x, valid_acc, label="Validation Accuracy")
plt.plot([60, 60], [0.2, 1.1], "k--", lw=1, alpha=0.7)
plt.ylim([0.2, 1.1])
plt.xticks((0, 50, 60, 100, 150, 200, 250), (0, 50, 60, 100, 150, 200, 250), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12)
plt.grid("on", linestyle="--", linewidth=0.5, alpha=0.5)
plt.show()


#
# Function to Plot Histograms
#

def plot_hist(df, idx, xstr, figname, xname, yrange=[0.75, 1], figsize=(9, 5)):
    res = df.iloc[idx]
    ta = res["acc"].values.tolist()
    ha = res["hgg_acc"].values.tolist()
    la = res["lgg_acc"].values.tolist()
    x = np.arange(len(idx)) * 1.5

    plt.figure(num=figname, figsize=figsize)
    plt.bar(x - 0.25, ta, width=0.25, label="Total")
    plt.bar(x, ha, width=0.25, label="HGG")
    plt.bar(x + 0.25, la, width=0.25, label="LGG")
    plt.ylim(yrange)
    plt.xticks(x, xstr, fontsize=12)
    plt.xlabel(xname, fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=11, ncol=3)
    plt.grid("on", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.show()


#
# Plot Metrics Histogram of Different Epoch Numbers
#

idx = [1, 2, 3, 4, 5, 0]
xstr = list(map(str, [30, 60, 100, 150, 200, 250]))
plot_hist(res_df, idx, xstr, "epoch_hist", "Epoch")


#
# Plot Metrics Histogram of Different Learning Rates
#

idx = [6, 7, 2, 8, 9, 10]
xstr = ["1e-3", "1e-4", "1e-5", "1e-3~1e-5", "1e-4~1e-6", "1e-5~1e-7"]
plot_hist(res_df, idx, xstr, "lr_hist", "Learning Rate")


#
# Plot Metrics Histogram of Different Initializers
#

idx = [19, 2, 20, 21, 22]
xstr = ["Orthognal", "Glorot Uniform", "Glorot Normal", "He Uniform", "He Normal"]
plot_hist(res_df, idx, xstr, "init_hist", "Initializer")


#
# Plot Metrics Histogram of Different Batch Normalization Momentums
#

idx = [2, 11, 12]
xstr = ["0.9", "0.95", "0.99"]
plot_hist(res_df, idx, xstr, "bn_hist", "Batch Normalization Momentum", figsize=(6, 5))


#
# Plot Metrics Histogram of Different Dropout Rates
#

idx = [29, 30, 31, 32, 33, 2, 34, 35, 36, 37]
xstr = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
plot_hist(res_df, idx, xstr, "dp_hist", "Dropout Rate")


#
# Plot Metrics Histogram of Different Optimizers
#

idx = [2, 15, 16]
xstr = ["Adam", "Adagrade", "SGD"]
plot_hist(res_df, idx, xstr, "opt_hist", "Optimizer", yrange=[0.5, 1], figsize=(6, 5))


#
# Plot Metrics Histogram of Different L2 Coefficients
#

idx = [17, 2, 18]
xstr = ["1e-5", "5e-5", "1e-4"]
plot_hist(res_df, idx, xstr, "l2_hist", "L2 Coefficient", figsize=(6, 5))


#
# Plot Metrics Histogram of Different Batch Size
#

idx = [28, 27, 26, 25, 24, 23, 2]
xstr = ["4", "6", "8", "10", "12", "14", "16"]
plot_hist(res_df, idx, xstr, "batch_hist", "Batch Size")
