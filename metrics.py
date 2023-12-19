import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from scipy.spatial.distance import directed_hausdorff

""" Loss Functions -------------------------------------"""
class BCEDiceLoss(tf.keras.losses.Loss):
    """Combines BCE and Dice Loss.

    Args:
        alpha (float): Weighting factor for BCE (default is 0.2).

    Attributes:
        alpha (float): Weighting factor for BCE.
        beta (float): Weighting factor for Dice Loss.
        epsilon (float): Small constant to avoid division by zero.
        bce_loss (BinaryCrossentropy): BCE loss.

    Methods:
        compute_dice(y_true, y_pred): Computes the Dice coefficient.
        call(y_true, y_pred): Computes combined BCE-Dice loss.
    """
    
    def __init__(self, alpha=0.2, **kwargs):
        """Initialize BCEDiceLoss."""
        super(BCEDiceLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = 1. - alpha
        self.epsilon = tf.keras.backend.epsilon()
        
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        
    def compute_dice(self, y_true, y_pred):
        """Compute the Dice coefficient."""
        y_true_f = K.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + self.epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.epsilon)
        return dice
    
    def call(self, y_true, y_pred):
        """Compute combined BCE-Dice loss."""
        bce_loss = self.bce_loss(y_true, y_pred)
        dice_loss = 1. - self.compute_dice(y_true, y_pred)
        bce_dice_loss = self.alpha * bce_loss + self.beta * dice_loss
        return bce_dice_loss
        

""" Metrics ----------------------------------------"""
def dice_score(y_true, y_pred):
    """
    Compute the Dice coefficient between the groud truth and prediction masks.

    Args:
        y_true (numpy.ndarray): Ground truth mask.
        y_pred (numpy.ndarray): Prediction masks.

    Returns:
        float: Dice coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    epsilon = tf.keras.backend.epsilon()

    intersection = sum(y_true_f * y_pred_f)
    y_sum = sum(y_true_f) + sum(y_pred_f)
    return (2. * intersection + epsilon) / (y_sum + epsilon)

def aji_score(y_true, y_pred):
    """
    Compute the Aggregated Jaccard Index (AJI) between the groud truth and prediction masks.

    Args:
        y_true (numpy.ndarray): Ground truth mask.
        y_pred (numpy.ndarray): Predicted mask.

    Returns:
        float: Aggregated Jaccard Index.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    epsilon = tf.keras.backend.epsilon()
  
    intersection = sum(y_true_f * y_pred_f)
    union = sum(y_true_f) + sum(y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)

def mean_hausdorff_distance(y_tests, y_preds):
    """
    Compute the mean Hausdorff distance between the groud truth and prediction masks.

    Args:
        y_tests (numpy.ndarray): Ground truth mask.
        y_preds (numpy.ndarray): Prediction mask.

    Returns:
        float: Mean Hausdorff distance.
    """
    assert y_tests.shape == y_preds.shape

    hds = [
        directed_hausdorff(np.squeeze(y_test), np.squeeze(y_pred))[0]
        for y_test, y_pred in zip(y_tests, y_preds)
    ]

    return np.mean(hds)
