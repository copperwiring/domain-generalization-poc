import os
import csv
import pandas as pd

base_path = "/shared-network/syadav/domain_images_random"
target_csv_path = "/home/syadav/srishtiy/domain-generalization/gt/gt.csv"

base_files = os.listdir(base_path)

label_list = []
for file in base_files:
    label = file.split('_')[0]
    label_list.append(label)

gt_df = pd.DataFrame(label_list)
gt_df.to_csv(target_csv_path, index=False, header=False)
