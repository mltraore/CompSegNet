import sys
import argparse
from data import DataGenerator, transform
from metrics import BCEDiceLoss
from model import *
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

# tf.config.experimental_run_functions_eagerly(True)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument('--train-folder', '-tf',  type=str, required=True)
   parser.add_argument('--validation-folder', '-vf',  type=str, required=True)
   parser.add_argument('--checkpoints-folder', '-cf',  type=str, default="checkpoints/ckpts")
   parser.add_argument('--optimizer', '-op', type=str, default='adam')
   parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
   parser.add_argument('--train-batch-size', '-trbs', type=int, default=4)
   parser.add_argument('--test-batch-size', '-tebs', type=int, default=1)
   parser.add_argument('--image-size', '-imsize', type=int, default=256)
   parser.add_argument('--epochs', '-eps', type=int, default=100)
   
   try:
       args = parser.parse_args()   
   except:
       parser.print_help()
       sys.exit(0)

   print('---------------------Welcome to CompSegNet-------------------')
   print('Author')
   print('Github: ********************')
   print('Email: *********************')
   print('CompSegNet model training details:')
   print('==================================')
   for i, arg in enumerate(vars(args)):
       print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
   
   # Data generators
   train_generator = DataGenerator(root_dir=args.train_folder,  
                                image_folder='images',
                                mask_folder='masks',
                                image_size=args.image_size,
                                batch_size=args.train_batch_size,
                                transform=transform,
                                shuffle=True)                    
                                
   val_generator = DataGenerator(root_dir=args.validation_folder,
                                image_folder='images',
                                mask_folder='masks',
                                image_size=args.image_size,
                                batch_size=args.test_batch_size,
                                transform=None,
                                shuffle=True)

   # Build model
   model = CompSegNet()
   model=model.build_graph(input_shape=(args.image_size, args.image_size, 3))
   model.summary()
   
   # Set up loss function
   loss = BCEDiceLoss()
   
   # Set up Optimizer
   if args.optimizer == 'adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, weight_decay=0.1)
   else:
      try:
          optimizer_class = getattr(tf.keras.optimizers, args.optimizer)
          optimizer = optimizer_class(learning_rate=args.learning_rate, weight_decay=0.1)
      except AttributeError:
          print(f"Error: Unknown optimizer '{args.optimizer}'")
          print("Please choose a valid optimizer or define your own.")
          sys.exit(0)
          
          
   # Callbacks 
   lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
       factor=0.1,
       patience=5,
       cooldown=5,
       min_lr=0.1e-6,
       verbose=1
   )                          
   early_stop = tf.keras.callbacks.EarlyStopping(
       monitor="val_loss",
       min_delta=0,
       patience=10,
       verbose=1,
       mode="auto",
       baseline=None,
       restore_best_weights=True,
       start_from_epoch=0
   )
   
   callbacks = [lr_reducer, early_stop]
   
   # Compile model 
   model.compile(
       optimizer=optimizer,
       loss=loss,
       metrics=['accuracy']
   )
   
   print('-------------Training CompSegNet------------')
   model.fit(
       train_generator,
       validation_data=val_generator,
       use_multiprocessing=True,
       workers=6, epochs=args.epochs,
       callbacks = callbacks)
   
   print('---------------Saving weights--------------')
   model.save_weights(args.checkpoints_folder)   
