import torch
import clip
from PIL import Image
import os

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_path = "/shared-network/syadav/real/The_Eiffel_Tower"
file = os.listdir("/shared-network/syadav/real/The_Eiffel_Tower")[0]
file_path = os.path.join(base_path, file)

image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

text = clip.tokenize(["This is a golden image", "This is an image of a tower", "This is a beautiful jewellery"])
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

