import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sprite_images import create_sprite
import csv, shutil, torch, pandas as pd, numpy as np
import clip
import glob
from tensorboard.plugins import projector
from shutil import copy2
from pathlib import Path
import tensorflow as tf

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
# Get working directory
PATH = os.getcwd()
LOG_DIR = PATH + "/tf_logs/embedding"  # path to the folder that we want to save the logs for Tensorboard


csv_domains = pd.read_csv("domains.csv", names=["class_names"])
domain_names = csv_domains["class_names"].to_list()

domain_embed = {}

with torch.no_grad():
    for each_domain in domain_names:
        text_tokens = clip.tokenize(["This is a photo of " + each_domain]).to(device)
        domain_embed[each_domain] = model.encode_text(text_tokens).float()

# Delete if directory exits and create new one
if os.path.exists(LOG_DIR) and os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
else:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Save embedding file
with open(os.path.join(LOG_DIR, 'text_embeddings.tsv'), 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for each_domain in domain_names:
        vector = domain_embed[each_domain].tolist()[0]
        writer.writerow(vector)

# Save labels
copy2('domains.tsv', LOG_DIR)

# Load the metadata file. Metadata consists your labels. This is optional.
# Metadata helps us visualize(color) different clusters that form t-SNE
metadata = os.path.join(LOG_DIR, 'domains.tsv')

# Save embeddings
# Source: https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

# The embedding variable, which needs to be stored
# Note this must a Variable not a Tensor!
sess = tf.compat.v1.Session()

embed_var = pd.read_csv(os.path.join(LOG_DIR, 'text_embeddings.tsv'))
embedding_var = tf.Variable(embed_var.to_numpy())

saver = tf.compat.v1.train.Saver([embedding_var])
with tf.compat.v1.Session() as sess:
  saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))


# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = "../../domains.tsv"

summary_writer = tf.summary.FileWriter(LOG_DIR)

projector.visualize_embeddings(summary_writer, config)



















# # Set up config.
# config = projector.ProjectorConfig()
# embedding = config.embeddings.add()
#
# # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
# embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
# embedding.tensor_path = "../../text_embeddings.tsv"
# embedding.metadata_path = "../../domains.tsv"
# projector.visualize_embeddings(LOG_DIR, config)
