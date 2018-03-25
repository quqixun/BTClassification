from __future__ import print_function

import os
import pandas as pd


parent_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(parent_dir, "test_logs")
model_dirs = os.listdir(input_dir)
models_num = len(model_dirs) + 1
res_list = []
for i in range(models_num):
    model_dir = "model" + str(i)
    print(model_dir)
    res_path = os.path.join(input_dir, model_dir, "valid_metrics.csv")

    if not os.path.isfile(res_path):
        continue

    df = pd.read_csv(res_path)
    df["model"] = model_dir
    res_list.append(df)
res_df = pd.concat(res_list)
res_df.to_csv("valid_res.csv", index=False)
