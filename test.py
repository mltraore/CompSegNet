import os
import sys
import argparse
import numpy as np
from natsort import natsorted
from prettytable import PrettyTable
from matplotlib import pyplot as plt

from model import *
from metrics import *
from tensorflow.keras.models import load_model

def load_test_set(set_test_path):
    """
    Load the test set images and masks.

    Args:
        set_test_path (str): Path to the test set directory containing 'images' and 'masks' subdirectories.

    Returns:
        tuple: A tuple containing two arrays - x_test (test images) and y_test (corresponding test masks).
    """
    test_image_list = natsorted(next(os.walk(os.path.join(set_test_path, 'images')))[2])
    test_mask_list = natsorted(next(os.walk(os.path.join(set_test_path, 'masks')))[2])

    assert len(test_image_list) == len(test_mask_list)
  
    x_test, y_test = [], []

    for iname, mname in zip(test_image_list, test_mask_list):
        image = plt.imread(os.path.join(set_test_path, 'images', iname))
        x_test.append(image)

        mask = plt.imread(os.path.join(set_test_path, 'masks', mname))
        mask = (mask > 0.5).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
        y_test.append(mask)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    return x_test, y_test


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument('--model-weights-save', '-ws', type=str, help="path to the save model weights", default='checkpoints/ckpts')
   parser.add_argument('--test-set', type=str, help="path to the test set containing images and labels", default='datasets/prepared_datasets/monuseg_2018/test')
   parser.add_argument('--image-size', '-imsize', type=int, default=256)
      
   try:
       args = parser.parse_args()   
   except:
       parser.print_help()
       sys.exit(0)
   
   print('---------------------Welcome to CompSegNet-------------------')
   print('Author')
   print('Github: ********************')
   print('Email: *********************')
   print('CompSegNet model prediction details:')
   print('==================================')
   for i, arg in enumerate(vars(args)):
       print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
   
   # Build model
   model = CompSegNet()
   model = model.build_graph(input_shape=(args.image_size, args.image_size, 3))
   
   # Load model weights
   model.load_weights(args.model_weights_save)
   
   # Load test images and labels
   test_imgs, test_lbls = load_test_set(args.test_set)
   
   # Predict
   predictions = model.predict(test_imgs/255.0, batch_size=1, verbose=1)
   predictions = (predictions > 0.5).astype(np.uint8)

   # Metric scores
   aji = aji_score(predictions, test_lbls)
   dice = dice_score(predictions, test_lbls)
   haus_dist = mean_hausdorff_distance(predictions, test_lbls)

   # Display metric scores
   table = PrettyTable(['Metric', 'Value'])
   table.add_row(['Aji', round(aji, 3)])
   table.add_row(['Dice', round(dice, 3)])
   table.add_row(['HD', round(haus_dist, 3)])
  
   print("Evaluation Metrics:")
   print(table)
  
