from PIL import Image
import matplotlib as plt
import numpy as np
import torch
from torchvision import transforms
from dpid import dpid
import sys
# setting path
from fastLayerDecomposition.Additive_mixing_layers_extraction import Hull_Simplification_determined_version
import time
import cv2
import os

to_PIL = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def get_palette_convex_hull(concatenated_image):
    # takes a concatenated image of the dataset as input and returns the color palette
    start=time.time()
    concatenated_image_normed = concatenated_image / 255.0
    palette_rgb = Hull_Simplification_determined_version(concatenated_image_normed, \
        "convexhull_vertices") * 255.0
    end=time.time()    
    M=len(palette_rgb)
    print("palette size: ", M)
    print("palette extraction time: ", end-start)
    palette_rgb = palette_rgb.reshape((1, -1, 3)).astype(np.float32)
    palette_rgb = cv2.cvtColor(palette_rgb, cv2.COLOR_BGR2RGB)
    palette_rgb_pil = Image.fromarray(palette_rgb.astype(np.uint8))
    t_palette_rgb = to_tensor(palette_rgb_pil)
    t_palette_rgb = torch.squeeze(torch.permute(t_palette_rgb, (2, 1, 0)))
    return t_palette_rgb, M

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


def quantize_images(folder_path, palette):
    # Create output folder
    output_folder = os.path.join(folder_path, '..', 'warped')
    os.makedirs(output_folder, exist_ok=True)

    # Load and quantize images
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            # Load image and get original size
            image = Image.open(os.path.join(folder_path, file)).convert("RGBA")
            width, height = image.size
            new_width, new_height = int(width / 16), int(height / 16)

            # Resize with nearest neighbor interpolation
            open_cv_image = np.array(image)

            Z = np.float32(open_cv_image[...,:3]).reshape((-1, 3))
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 6
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res = res.reshape((open_cv_image.shape[0], open_cv_image.shape[1], 3))
            open_cv_image[..., :3] = res

            small_img = dpid(open_cv_image, 10, lamb=2.5)
            alpha_layer = small_img[..., 3] 
            alpha_layer[alpha_layer < 125.0] = 0
            alpha_layer[alpha_layer >= 125.0] = 255
            small_img[..., 3] = alpha_layer
            small_img = Image.fromarray(small_img).convert("RGBA")
            #small_img = image.resize((new_width, new_height), Image.NEAREST)

            # Calculate Background mask
            #background_mask = np.array(small_img)[..., 3] == 0
            
            ## convert to RGB with white background
            #new_image = Image.new("RGBA", small_img.size, "WHITE")
            #new_image.paste(small_img, mask=small_img)
            #new_image = new_image.convert("RGB")

            ## Convert image to numpy array and reshape to a 2D array
            #img = np.array(new_image) / 255.0

            ## Quantize colors using palette
            #palette = np.array(palette)
            #img = np.array([palette[np.argmin(np.linalg.norm(palette - pixel, axis=1))] \
            #    for pixel in np.resize(img, (new_width * new_height, 3))])

            ## Resize back to original size with nearest neighbor interpolation
            #img = np.resize(img, (new_width, new_height, 3))
            #img_rgba = np.ones((new_width, new_height, 4))
            #img_rgba[..., :3] = img
            #img_rgba[background_mask, 3] = 0
            #img_rgba = Image.fromarray(np.uint8(img_rgba * 255.0)).convert("RGBA")
            #img_rgba = img_rgba.resize((width, height), Image.NEAREST)

            # Save image
            img_rgba = small_img.resize((width, height), Image.NEAREST)
            img_rgba.save(os.path.join(output_folder, file))

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
