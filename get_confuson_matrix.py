import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from plot_confusion_matrix import plot_multi_label_cm

gt_path = "/home/syadav/srishtiy/domain-generalization/gt/gt.csv"
pred_path = "/home/syadav/srishtiy/domain-generalization/pred/pred_domain.csv"

gt_file = pd.read_csv(gt_path, header=None)
gt_values = gt_file.values
gt = []
for each_gt in gt_values:
    gt.append(each_gt[0])


pred_file = pd.read_csv(pred_path, header=None)
pred_values = pred_file.values
pred = []
for each_pred in pred_values:
    pred.append(each_pred[0])

# Save Confusion Matrix
cm = confusion_matrix(gt, pred)

target_names = ['quickdraw',  'clipart', 'infograph', 'real', 'sketch']
plot_multi_label_cm(cm, target_names)
