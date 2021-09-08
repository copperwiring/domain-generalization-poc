import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv, shutil, torch, pandas as pd
import clip
from tensorboard.plugins import projector
from shutil import copy2
from pathlib import Path


# device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()

csv_classes = pd.read_csv("real_classes.csv", names=["class_names"])
real_classes = csv_classes["class_names"].to_list()

domain_embed = {}

with torch.no_grad():
    for each_class in real_classes:
        text_tokens = clip.tokenize(["This is a photo of " + each_class]).to(device)
        domain_embed[each_class] = model.encode_text(text_tokens).float()

# Get working directory
PATH = os.getcwd()
LOG_DIR = PATH + "tf_logs/embedding"  # path to the folder that we want to save the logs for Tensorboard

# Delete if directory exits
if os.path.exists(LOG_DIR) and os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR)

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
# Save embedding file
with open(os.path.join(LOG_DIR, 'text_embeddings.tsv'), 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for each_class in real_classes:
        vector = domain_embed[each_class].tolist()[0]
        writer.writerow(vector)

# Save labels
copy2('real_classes.tsv', LOG_DIR)

# Load the metadata file. Metadata consists your labels. This is optional.
# Metadata helps us visualize(color) different clusters that form t-SNE
metadata = os.path.join(LOG_DIR, 'real_classes.tsv')

# Set up config.

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.tensor_path = 'text_embeddings.tsv'
embedding.metadata_path = 'real_classes.tsv'
projector.visualize_embeddings(LOG_DIR, config)

#
# text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)
#
#
# classes_and_labels = []
# each_list = []
#
# for i, image in enumerate(original_images):
#     each_prob = top_probs[i]
#
#     # import pdb; pdb.set_trace()
#     class_name = [real_classes[index] for index in top_labels[i].numpy()]
#
#     each_list.append([each_prob.numpy()[0], class_name[0]])
#
# print(each_list)
