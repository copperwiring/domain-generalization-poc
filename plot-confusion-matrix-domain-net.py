import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import csv, shutil, torch, pandas as pd, numpy as np
import clip
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_cm_matrix


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
# Get working directory
PATH = os.getcwd()

base_path = "/shared-network/syadav/all_domain_few_img"
image_input = []
original_images = []

# Get images
# Check os.listdir(base_path)
for file in os.listdir(base_path):
    file_path = os.path.join(base_path, file)
    image = Image.open(file_path).convert("RGB")
    original_images.append(image)
    image_input.append(preprocess(image))

# Get text embeds
csv_domains = pd.read_csv("domains.tsv", names=["class_names"])
domain_names = csv_domains["class_names"].to_list()

domain_embed = {}

# Text Embedding
text_inputs = torch.cat([clip.tokenize(f"This is a photo of a {c}") for c in domain_names]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

# Print the result
for i, image in enumerate(original_images):
    print(top_probs[i])
    print(domain_names[index] for index in top_labels[i].numpy())

# gt =
# pred =

# Save Confusion Matrix
cm = confusion_matrix(gt, pred)
plot_cm_matrix(gt, pred, cm)


