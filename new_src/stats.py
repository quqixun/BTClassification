import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.getcwd())
test_logs_dir = os.path.join(parent_dir, "test_logs")

test_logs = os.listdir(test_logs_dir)
test_logs_paths = []
for test_log in test_logs:
    test_logs_paths.append(os.path.join(test_logs_dir, test_log))

df = pd.concat(map(pd.read_csv, test_logs_paths))

acc = df["acc"].values
hgg_acc = df["hgg_acc"].values
lgg_acc = df["lgg_acc"].values

loss = df["loss"].values
hgg_loss = df["hgg_loss"].values
lgg_loss = df["lgg_loss"].values

hgg_precision = df["hgg_precision"].values
lgg_precision = df["lgg_precision"].values
hgg_recall = df["hgg_recall"].values
lgg_recall = df["lgg_recall"].values

accs = [acc, hgg_acc, lgg_acc]
losses = [loss, hgg_loss, lgg_loss]
prs = [hgg_precision, hgg_recall, lgg_precision, lgg_recall]

plt.figure()
plt.title("Prediction Accuracy for Different Data", fontsize=14)
plt.boxplot(accs, 0, "k.", labels=["Total", "HGG", "LGG"])
plt.grid("on", linestyle="--", linewidth=0.5)
axes = plt.gca()
axes.set_ylim([0.5, 1.02])
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure()
plt.title("Prediction Cross Entropy for Different Data", fontsize=14)
plt.boxplot(losses, 0, "k.", labels=["Total", "HGG", "LGG"])
plt.grid("on", linestyle="--", linewidth=0.5)
axes = plt.gca()
axes.set_ylim([0, 1])
plt.ylabel("Cross Entropy", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure()
plt.title("Precision and Recall for HGG and LGG", fontsize=14)
plt.boxplot(prs, 0, "k.", labels=["HGG\nPrecision", "HGG\nRecall",
                                  "LGG\nPrecision", "LGG\nRecall"])
plt.grid("on", linestyle="--", linewidth=0.5)
axes = plt.gca()
axes.set_ylim([0.5, 1.02])
plt.ylabel("Precision or Recall", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
