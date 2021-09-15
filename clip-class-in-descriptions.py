import torch
from PIL import Image
import os
import clip
import os
import skimage
import numpy as np
import torch

from PIL import Image
from collections import OrderedDict

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
#
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

base_path = "/shared-network/syadav/real"
dir_list = os.listdir(base_path)

# images in skimage to use and their textual descriptions
descriptions_list = [
    "This is an image of a beautiful eiffel tower",
    "This is an image of a large wall called The Great Wall of China",
    "This is a popular painting of Mona Lisa",
    "This is a large aircraft carrier",
    "This is a large picture of an aeroplane"
]
original_images = []
images = []
texts = []

for i in range(0, 5):
    base_dir_path = os.path.join(base_path, dir_list[i])

    file_list = os.listdir(base_dir_path)

    first_file = file_list[0];
    file_name = os.path.splitext(first_file)[0]
    file_path = os.path.join(base_dir_path, first_file)

    image = Image.open(file_path).convert("RGB")

    original_images.append(image)
    images.append(preprocess(image))

# Build features
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

# Calculate cosine similarity
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

for y in range(similarity.shape[0]):
    for x in range(similarity.shape[1]):
        print(similarity[x, y], end=' ')
    print("\n")

# print(similarity)
