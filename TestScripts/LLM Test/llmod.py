# Use a pipeline as a high-level helper
from transformers import pipeline
import skimage
import numpy as np
from PIL import Image

image = Image.open(r"C:\Users\drago\Downloads\checkerboard_pi_images\l.jpg").convert("RGB")

pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device='cuda')

predictions = pipe(
    image,
    candidate_labels=["Screw"],
)

print(predictions)