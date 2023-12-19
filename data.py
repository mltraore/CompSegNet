import os
import cv2
import numpy as np
import tensorflow as tf
from natsort import natsorted
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from albumentations import (Compose, GaussianBlur, MedianBlur, CLAHE, Emboss,
                            RandomBrightnessContrast, RandomCrop, RandomRotate90,
                            GaussNoise, HorizontalFlip, VerticalFlip)


def transform():
    """
    Create and return a list of image augmentation operations to be applied during the training process.

    Returns:
        list: A list of image augmentation operations to be applied on images.
    """

    return  Compose([
                GaussianBlur(always_apply=False, p=0.25, blur_limit=(3, 7), sigma_limit=(0.0, 0)),
                MedianBlur(always_apply=False, p=0.25, blur_limit=(3, 7)),
                CLAHE(always_apply=False, p=0.25, clip_limit=(1, 3), tile_grid_size=(4, 4)),
                Emboss(always_apply=False, p=0.25, alpha=(0.39, 0.45), strength=(0.2, 2.04)),
                RandomBrightnessContrast(always_apply=False, p=0.25, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
                RandomCrop(always_apply=False, p=0.25, height=220, width=220),
                RandomRotate90(always_apply=False, p=0.25),
                GaussNoise(always_apply=False, p=0.25, var_limit=(92.11, 149.67), per_channel=True, mean=0.0),
                HorizontalFlip(always_apply=False, p=0.25),
                VerticalFlip(always_apply=False, p=0.25),
            ])


class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for training process using image and mask data.

    Args:
        root_dir (str): Root directory path.
        image_folder (str): Folder name containing the images.
        mask_folder (str): Folder name containing the masks.
        image_size (int, optional): Size of the images. Defaults to 256.
        batch_size (int, optional): Batch size. Defaults to 4.
        transform (callable, optional): Image transformation function. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """

    def __init__(self,
                 root_dir,
                 image_folder,
                 mask_folder,
                 image_size=256,
                 batch_size=4,
                 transform=None,
                 shuffle=True):
        super(DataGenerator, self).__init__()

        self.path = root_dir
        self.image_names = natsorted(next(os.walk(os.path.join(root_dir, image_folder)))[2])
        self.mask_names = natsorted(next(os.walk(os.path.join(root_dir, mask_folder)))[2])

        self.image_size = image_size
        self.batch_size = batch_size

        self.currentIndex = 0
        self.indexes = None

        self.transform = transform

        self.shuffle = True

        self.on_epoch_end()


    def __len__(self):
        """
        Get the number of batches in the dataset.

        Returns:
            int: The number of batches in the dataset.
        """
        
        return int(np.ceil(len(self.image_names) / self.batch_size))


    def on_epoch_end(self):
        """
        Shuffle the training set at the end of each each epoch (if shuffle == True)
        """

        if self.shuffle:
           self.image_names, self.mask_names = shuffle(self.image_names, self.mask_names)


    def read_image_mask(self, image_name, mask_name, path):
        """
        Read and preprocess an image and its corresponding mask.

        Args:
            image_name (str): The filename of the image.
            mask_name (str): The filename of the mask.
            path (str): The root path of the images and masks.

        Returns:
            tuple: A tuple containing the preprocessed image and mask arrays.
        """

        image_path = path + '/images/'
        mask_path = path + '/masks/'

        image = plt.imread(os.path.join(image_path, image_name)).astype(np.uint8)
        if image.shape[2] == 4:
           image = image[:, :, :3]

        mask = plt.imread(os.path.join(mask_path, mask_name))
        mask = (mask > 0.5).astype(np.uint8)

        return image, mask


    def __getitem__(self, index):
        """
        Generate one batch of preprocessed data for the given index.

        Args:
            index (int): The index of the batch.

        Returns:
            tuple: A tuple containing the preprocessed input (X) and target (y) arrays for the batch.
        """

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes = self.image_names[start:end]

        im_in_batch = len(indexes)

        X = np.zeros((im_in_batch, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.zeros((im_in_batch, self.image_size, self.image_size, 1), dtype=np.float32)

        for i, sample_id in enumerate(indexes):

            image, mask = self.read_image_mask(self.image_names[index * self.batch_size + i],
                                               self.mask_names[index * self.batch_size + i],
                                               self.path)

            if self.transform:
               transformed = self.transform()(image=image, mask=mask)
               image_trans = transformed['image']
               mask_trans = transformed['mask']
               
               if image_trans.shape[0] < 256:
                  image_trans = cv2.resize(image_trans, (256, 256))
                  mask_trans = cv2.resize(mask_trans, (256, 256))
                  
               X[i, ...] = image_trans / 255.0
               y[i, ...] = np.expand_dims(mask_trans, -1)
            
            elif not self.transform and self.batch_size == 1:
               return image.reshape(1, image.shape[0], image.shape[1], 3) / 255.0, \
                      mask.reshape(1, mask.shape[0], mask.shape[1], 1)

        return X, y

if __name__ == "__main__":
   train_generator = DataGenerator(root_dir='datasets/prepared_datasets/monuseg_2018/train',  
                                image_folder='images',
                                mask_folder='masks',
                                image_size=256,
                                batch_size=4,
                                transform=transform,
                                shuffle=True)
   
   train_sample_image , train_sample_label = train_generator.__getitem__(2)   
   print(train_sample_image.shape , train_sample_label.shape)                     
      
   val_generator = DataGenerator(root_dir='datasets/prepared_datasets/monuseg_2018/validation',
                                image_folder='images',
                                mask_folder='masks',
                                image_size=256,
                                batch_size=1,
                                transform=None,
                                shuffle=True)
                                
   val_sample_image , val_sample_label = val_generator.__getitem__(2)   
   print(val_sample_image.shape , val_sample_label.shape)
