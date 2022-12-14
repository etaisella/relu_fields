from PIL import Image
import matplotlib as plt
import numpy as np
import torch
from torchvision import transforms
import cv2
import os

to_PIL = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def get_palette(concatenated_image, num_clusters):
    # takes a concatenated image of the dataset as input and returns the color palette
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = num_clusters
    _, _, center = cv2.kmeans(concatenated_image, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = center.reshape((1, -1, 3))
    center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
    center_pil = Image.fromarray(center.astype(np.uint8))
    t_centers = to_tensor(center_pil)
    t_centers = torch.squeeze(torch.permute(t_centers, (2, 1, 0)))
  
    return t_centers

def concatenate_images(input_folder):
    # Create an empty image with the same width as the input images and
    # enough height to fit all the images
    images = [Image.open(os.path.join(input_folder, f)).convert("RGB") for f in os.listdir(input_folder)]
    width, _ = images[0].size
    height = sum([im.size[1] for im in images])
    result = Image.new('RGB', (width, height))

    # Concatenate the images into the result image
    y_offset = 0
    for im in images:
        result.paste(im, (0, y_offset))
        y_offset += im.size[1]

    # Return the result image
    result = np.array(result)
    open_cv_image = result[:, :, ::-1].copy() 
    Z = np.float32(open_cv_image.reshape((-1,3)))
    return Z
