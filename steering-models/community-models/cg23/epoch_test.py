# -----------------------------------------------------------------------------------------
# Challenge #2 - epoch_test.py - run script
# -----------------------------------------------------------------------------------------

'''
Builds, Trains and Tests the Steering Model using a Data Generator
*Note: k-fold training is currently commented out
Original By: dolaameng Revd: cgundling
'''

from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
import os
from sklearn.cross_validation import KFold
from os import path
from collections import defaultdict
import time
import random

from epoch_data import *
from epoch_model import *

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Main Program
# ----------------------------------------------------------------------------------------
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--model', type=str, help='model to evaluate, current list: {cnn}')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
    parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    args = parser.parse_args()

    # Model and data gen args
    # ---------------------------------------------------------------------------------
    dataset_path = args.dataset
    model_name = args.model
    image_size = (args.resized_image_height, args.resized_image_width)
    camera = args.camera
    camera_train = 'center','right','left'
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    weights_path = None   #'weights_HMB_1.hdf5'

    # Data paths
    # ---------------------------------------------------------------------------------
    # build model and train it
    steering_log = path.join(dataset_path, 'steering.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    # Model build
    # ---------------------------------------------------------------------------------
    model_builders = {
                     'V3': (build_InceptionV3, preprocess_input_InceptionV3, exact_output) 
                     , 'cnn': (build_cnn, normalize_input, exact_output)}

    if model_name not in model_builders:
        raise ValueError("unsupported model %s" % model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    model = model_builder(image_size,weights_path)
    print('model %s built...' % model_name)

    # Setup the k-fold cross validation
    # ---------------------------------------------------------------------------------
    data = data_kfold(steering_log)
    timebreak_start, timebreak_end, unique_times = kfold_cross(data)
    
    # For averaging k-folds
    yfull_test = []
        
    # k-fold setup 
    nfolds = 10
    random_state = 51
    #kf = KFold(len(unique_times), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
        
    # Use epoch_data.py to generate training data
    # ---------------------------------------------------------------------------------
    #for train_times, test_times in kf:
    if not weights_path:
        unique_list_train = unique_times #[int(unique_times[i]) for i in train_times]
        time_list_train = []
        time_list_test = []
        for j in unique_list_train:
            time_list_train.append(range(timebreak_start[j-1],timebreak_end[j-1]+1))
                    
        time_list_train = [val for sublist in time_list_train for val in sublist]
        train_generator = data_generator(steering_log=steering_log,
                            image_log=image_log,
                            image_folder=camera_images,
                            unique_list=time_list_train,
                            gen_type='train',
                            camera=camera_train,
                            batch_size=batch_size,
                            image_size=image_size,
                            # Round 2 Complete Dataset
                            timestamp_start=14774295160,
                            timestamp_end=14794265723,
                            shuffle=True,
                            preprocess_input=input_processor,
                            preprocess_output=output_processor)
 
        # Use epoch_data.py to generate validation data
        # -----------------------------------------------------------------------------
        unique_list_test = [68,69,70] #[int(unique_times[i]) for i in test_times]
        #for j in unique_list_test:
        #    time_list_test.append(range(timebreak_start[j-1],timebreak_end[j-1]+1))
                
        time_list_test = range(14774405347-1600,14774405347+1) #[val for sublist in time_list_test for val in sublist]
        val_generator = data_generator(steering_log=steering_log,
                            image_log=image_log,
                            image_folder=camera_images,
                            unique_list=time_list_test,
                            gen_type='val',
                            camera=camera,
                            batch_size=32, 
                            image_size=image_size,
                            # Round 2 Complete Dataset
                            timestamp_start=14774405347-1600,
                            timestamp_end=14774405347,
                            shuffle=False,
                            preprocess_input=input_processor,
                            preprocess_output=output_processor)
    
        num_fold += 1            
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Train TimeBreaks: ', unique_list_train)
        print('Test TimeBreaks: ', unique_list_test)

        # Training the model - with EarlyStopping, ModelCheckpoint
        # ------------------------------------------------------------------------------
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), 
                    ModelCheckpoint(filepath=os.path.join('weights_HMB_' + str(num_fold) + '.hdf5'), 
                    monitor='val_loss', verbose=0, save_best_only=True)]
        model.fit_generator(train_generator, samples_per_epoch=64000, nb_epoch=nb_epoch,verbose=1,
                    callbacks=callbacks, validation_data=val_generator,nb_val_samples=1600)

        print('kfold model successfully trained...')
                

    # Use epoch_data.py to generate test data
    # Test set here is same as validation data - seperate script was used for final tests
    # -----------------------------------------------------------------------------------
    time_list_test = range(14774405347-1600,14774405347+1)
    test_generator = data_generator(steering_log=steering_log,
                        image_log=image_log, 
                        image_folder=camera_images,
                        unique_list=time_list_test,
                        gen_type='test',
                        camera=camera,
                        batch_size=1600,
                        image_size=image_size,
                        # Round 2 Testing
                        timestamp_start=14774405347-1600,
                        timestamp_end=14774405347,
                        shuffle=False,
                        preprocess_input=input_processor,
                        preprocess_output=output_processor)
       
    print('Testing for this fold')

    test_x, test_y = test_generator.next()
    print('test data shape:', test_x.shape, test_y.shape)

    # Store test predictions
    yhat = model.predict(test_x)
    #yfull_test.append(yhat)
        
    # Merge the test predictions for each fold
    #print('Made it to Final Test')
    #test_res = combine_folds(yfull_test, nfolds)

    # Use dataframe to write results, calculate RMSE and plot actual vs. predicted steerings
    # ------------------------------------------------------------------------------------ 
    df_test = pd.read_csv('output1.csv',usecols=['frame_id','steering_angle','pred'],index_col = None)
    df_test['steering_angle'] = test_y
    df_test['pred'] = yhat # test_res
    df_test.to_csv('output2.csv')
        
    # Calculate RMSE
    # ------------------------------------------------------------------------------------
    sq = 0
    mse = 0
    for j in range(test_y.shape[0]):
        sqd = ((yhat[j]-test_y[j])**2)
        sq = sq + sqd
    print(sq)
    mse = sq/1600
    rmse = np.sqrt(mse)
    print("model evaluated RMSE:", rmse)

    # Plot the results
    # ------------------------------------------------------------------------------------
    plt.figure(figsize = (32, 8))
    plt.plot(test_y, 'r.-', label='target')
    plt.plot(yhat, 'b.-', label='predict')
    plt.legend(loc='best')
    plt.title("RMSE Evaluated on 1600 TimeStamps: %.4f" % rmse)
    plt.show()
    model_fullname = "%s_%d.png" % (model_name, int(time.time()))
    plt.savefig(model_fullname)


if __name__ == '__main__':
    main()
