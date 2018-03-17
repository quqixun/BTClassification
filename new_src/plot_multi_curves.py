from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.getcwd())
models_dir = os.path.join(parent_dir, "models")
csv_name = "learning_curve.csv"


def plot_multi_curves(dfs, labels,
                      figure_name="learning_curves",
                      alphas=[0.6, 0.6, 0.6, 1.0]):

    metrics = ["acc", "loss", "val_acc", "val_loss"]
    for metric in metrics:
        plt.figure(num=figure_name + "_" + metric)
        for df, label, alpha in zip(dfs, labels, alphas):
            curve = df[metric].values.tolist()
            num = len(curve)
            x = np.arange(1, num + 1)
            if alpha == 1.0:
                plt.plot(x, curve, color="k", label=label, alpha=alpha)
            else:
                plt.plot(x, curve, label=label, alpha=alpha)

        if "loss" in metric:
            legend_loc = 1
            ylim = [0.0, 2.5]
            ylabel = "Loss"
        else:
            legend_loc = 4
            ylim = [0.4, 1.0]
            ylabel = "Accuracy"
        plt.ylim(ylim)
        plt.xlim([0, num])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.legend(fontsize=16, loc=legend_loc, ncol=2)
        plt.grid("on", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
    return


lr_models_names = ["model-test-lr-2", "model-test-lr-3",
                   "model-test-lr-4", "model-afm-max-adam-5-5"]

lr_labels = ["1e-3", "1e-4", "1e-5", "1e-3~1e-4~1e-5"]
lr_dfs = [pd.read_csv(os.path.join(models_dir, model_name, csv_name))
          for model_name in lr_models_names]
plot_multi_curves(lr_dfs, lr_labels, "lr")


bs_models_names = ["model-test-bs-1", "model-test-bs-2",
                   "model-test-bs-3", "model-afm-max-adam-5-5"]

bs_labels = ["4", "8", "12", "16"]
bs_dfs = [pd.read_csv(os.path.join(models_dir, model_name, csv_name))
          for model_name in bs_models_names]
plot_multi_curves(bs_dfs, bs_labels, "bs")
