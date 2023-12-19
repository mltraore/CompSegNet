import os
import glob
import shutil
import argparse
import numpy as np
from skimage import io, transform
from scipy.io import loadmat
from patchify import patchify
from natsort import natsorted
from sklearn.model_selection import train_test_split

import stainNorm_Macenko

import warnings
warnings.filterwarnings('ignore')

"""
Prepares datasets for nuclei segmentation tasks

Notes: 
  * Color normalization was applied to MoNuSeg 2018 and CPM 2017 datasets only.
  * Carefully select the reference image for color normalization as it impacts results.  
  * Inspect images after normalization for potential artifacts.
  * Consider manually selecting validation images to maintain data distribution.
"""

def read_and_normalize_image(image_path, target_img, normalizer):
    """
    Reads and color-normalizes a nuclei segmentation image.

    Args:
        image_path (str): Path to the image file.
        target_img (np.ndarray): Target image used for color normalization.
        normalizer (stainNorm_Macenko.Normalizer): Color normalization object.

    Returns:
        np.ndarray: The color-normalized image.
    """
    img = io.imread(image_path)
    resized_img = transform.resize(
        img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), anti_aliasing=True)
    normalized_img = normalizer.transform(resized_img)
    return normalized_img

def read_and_resize_label(label_path):
    """
    Reads and resizes a segmentation label.

    Args:
        label_path (str): The path to the label file (MAT or image).
    Returns:
        np.ndarray: The resized label.
    """
    
    if label_path.split('.')[-1] == 'mat':
        label = loadmat(label_path)['inst_map']
    else:
        label = io.imread(label_path)

    resized_label = transform.resize(
        label, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), anti_aliasing=True)
    binary_label = (resized_label > 0).astype(np.uint8)
    return binary_label

def save_patch(patch, save_path):    
    """
    Saves an image patch to the specified file path.

    Args:
        patch (np.ndarray): Image patch to be saved.
        save_path (str): File path to save the image patch.
    """
    io.imsave(save_path, patch)

def prepare_images(split_dirs):
    """
    Prepares images and labels by normalizing, patching, and saving to a new directory.

    Args:
        split_dirs (list): List of dataset split directories (e.g., train and test).
    """
    target_img = io.imread(REF_IMAGE_PATH)
    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(target_img)

    img_patch_no = 1
    lbl_patch_no = 1

    for split in split_dirs:
        image_list = natsorted(glob.glob(f'{DATASET_PATH}/{split}/images/*'))
        label_list = natsorted(glob.glob(f'{DATASET_PATH}/{split}/labels/*'))

        os.makedirs(os.path.join(PREPARED_DATASET_PATH, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(PREPARED_DATASET_PATH, split, 'masks'), exist_ok=True)

        for im in image_list:
            normalized_img = read_and_normalize_image(
                im, target_img, normalizer)
            img_patches = patchify(
                normalized_img, (PATCH_SIZE, PATCH_SIZE, 3), step=STEP_SIZE)
            img_patches = np.squeeze(img_patches)

            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    img_patch = img_patches[i, j]
                    save_path = os.path.join(
                        PREPARED_DATASET_PATH, split, 'images', f'{img_patch_no}.png')
                    save_patch(img_patch, save_path)
                    img_patch_no += 1
        
        for lbl in label_list:
            label = read_and_resize_label(lbl)
            label_patches = patchify(
                label, (PATCH_SIZE, PATCH_SIZE), step=STEP_SIZE)
            label_patches = np.squeeze(label_patches)

            for i in range(label_patches.shape[0]):
                for j in range(label_patches.shape[1]):
                    label_patch = label_patches[i, j]*255
                    save_path = os.path.join(
                        PREPARED_DATASET_PATH, split, 'masks', f'{lbl_patch_no}.png')
                    save_patch(label_patch, save_path)
                    lbl_patch_no += 1
                    
def create_validation(train_image_path, train_label_path, validation_image_path, validation_label_path):
    """
    Creates a validation set from the training set.

    Args:
        train_image_path (str): Path to the training set images.
        train_label_path (str): Path to the training set labels.
        validation_image_path (str): Path to save the validation set images.
        validation_label_path (str): Path to save the validation set labels.
    """
    os.makedirs(validation_image_path, exist_ok=True)
    os.makedirs(validation_label_path, exist_ok=True) 
     
    image_list = os.listdir(train_image_path)
   
    _, val_list = train_test_split(image_list, test_size=VAL_SIZE, random_state=RANDOM_STATE)
   
    for img_name in val_list:
        src_image_path = os.path.join(train_image_path, img_name)
        dest_image_path = os.path.join(validation_image_path, img_name)
        shutil.move(src_image_path, dest_image_path)
        
        lbl_name = img_name
        src_label_path = os.path.join(train_label_path, lbl_name)
        dest_label_path = os.path.join(validation_label_path, lbl_name)
        shutil.move(src_label_path, dest_label_path)       
          

if __name__ == "__main__": 
   parser = argparse.ArgumentParser(description="Dataset configuration")
   
   parser.add_argument('--dataset-path', '-dpath', type=str, default='datasets/monuseg_2018')
   parser.add_argument('--image-size', '-imsize', type=int, help="Image size (width/height)", default=1000)
   parser.add_argument('--validation-size', '-vsize', type=float, help="Validation size (%) (from train set)", default=0.1)          
   parser.add_argument('--reference-image', '-r', type=str, help="Path to the target image for color normalization", default='datasets/monuseg_2018/train/images/1.tif')
   parser.add_argument('--prepared-data-path', '-ppath', type=str, default='datasets/prepared_datasets/monuseg_2018')
   
   try:
       args = parser.parse_args()   
   except:
       parser.print_help()
       sys.exit(0)
   
   RANDOM_STATE = 42
   DATASET_PATH = args.dataset_path
   IMG_SIZE = args.image_size
   VAL_SIZE = args.validation_size
   REF_IMAGE_PATH = args.reference_image
   PREPARED_DATASET_PATH = args.prepared_data_path
   
   TARGET_IMG_SIZE = 512 if IMG_SIZE < 1000 else 1024
   PATCH_SIZE = 256
   STEP_SIZE = 256
   
   SPLIT_DIRS = ['train', 'test']

   # Process images 
   prepare_images(SPLIT_DIRS)
   
   # Create the validation set
   create_validation(train_image_path = os.path.join(PREPARED_DATASET_PATH, 'train', 'images'), 
                      train_label_path = os.path.join(PREPARED_DATASET_PATH, 'train', 'masks'), 
                      validation_image_path = os.path.join(PREPARED_DATASET_PATH, 'validation', 'images'),
                      validation_label_path = os.path.join(PREPARED_DATASET_PATH, 'validation', 'masks'))
