# coding: utf-8

import tensorflow as tf
import numpy as np
import scipy.io as sio
#import pandas as pd
import sys
import os
import time
import rmse as rmse
from utils import *
from model import *
from metrics import *
#from resnet_model import *
'''
# set the GPU parameters
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement = True
config.log_device_placement = True
sess = tf.Session(config=config)
KTF.set_session(sess)
'''

# set the network parameters
lr = 3e-5
#epochs = 30  
epochs = 15  
batch_size = 256
#batch_size = 64 
key = 0
p1 = 0
p2 = 1
timestep = 1;

# set the output path 
path_result = '../output/prediction'
path_model  = '../output/model'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model ) is False:
    os.mkdir(path_model)

root_path = os.getcwd().replace("\\","/") 
X_path = root_path + '/../input/SST_daily_global.mat'
Y_path = root_path + '/../input/Nino3.4_OISST_19820101-20161231.mat'
Z_path = root_path + '/../input/map_sst_1982-2016.mat'

X = load_data(X_path,'sst_daily')
X = np.asarray(X)
X = X[:,110:290,30:150]
Y = load_data(Y_path, 'nino34_daily')
micmap = np.array(load_data(Z_path, 'map_sst'))

varname = ["SST"]

for i in range(1):
    model_name = 'ResNet50_' + varname[i] + '_1deg_mic'
    print(model_name)

    X, Y, scalar = preprocess_data(X, Y, split_ratio=0.8)
    X = np.asarray(X)

    #day = [30,60,90,120,150,180,210,240,270,300,330,360]
    #day = [210,240,270]
    day = [210]
    for shift in day: 
        lead = int(shift/timestep)
        i = int(shift/30)-1
        dum = np.squeeze(micmap[:,:,i])
        dum = np.tile(dum,(len(X),1,1))
        X = X*dum
        print(X.shape)
        train_X, extra_train_X, train_Y, test_X, extra_test_X, test_Y = create_st_dataset(X, Y, predict_day=lead, split_ratio=0.8)
        print("train X shape: {}".format(train_X.shape))
        print("extra train X shape: {}".format(extra_train_X.shape))
        print("train Y shape: {}".format(train_Y.shape))

        #for it in range(1,11):
        for it in range(1,2):
            print("==>The current cycle for train: {}".format(it))
            ##time_start=time.time()
            if key == 0:
               model = Small_ResNet(train_X[0].shape, None)
            else:
               model = Small_ResNet(train_X[0].shape, extra_train_X[0].shape)
            ##time_end=time.time() 
            ##print('totally time cost',time_end-time_start)
 
            adam = tf.keras.optimizers.Adam(lr)
            model.compile(optimizer=adam, loss='mse', metrics=[rmse.rmse])

            model.summary()
            tf.keras.utils.plot_model(model, to_file=model_name+'.png', show_shapes=True, show_layer_names=True)

            hyperparams_name = 'model_name{}.predict_day{}.lr{}.time{}'.format(model_name, shift, lr, it)
            # set parameter
            fname_param = os.path.join(path_model, 'ck.{}.h5'.format(hyperparams_name))
            print(hyperparams_name)
            print(fname_param)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_square_error', patience=2, mode='min')
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fname_param, monitor='val_root_mean_square_error', verbose=0, save_best_only=True, mode='min')

            if os.path.exists(fname_param):
                #model.load_weights(fname_param)
            	print("do not load weights...")
            
            print("==>Training model...")

            if key == 0:
               history = model.fit(train_X,  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(test_X,test_Y), callbacks=[model_checkpoint])
            else:
               history = model.fit([train_X,extra_train_X],  train_Y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=([test_X, extra_test_X],test_Y), callbacks=[model_checkpoint])

            # set parameter
            #model.save_weights(fname_param, overwrite=True)

            print("==>Testing model...")
            if key == 0:
               score = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)
            else:
               score = model.evaluate([test_X,extra_test_X], test_Y, batch_size=batch_size, verbose=1)
            print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (score[0], score[1], score[1] * (scalar._max - scalar._min) / 2.))

            print("==>Predicting model...")
            if key == 0:
               y_hat_norm = model.predict(test_X, batch_size=batch_size, verbose=0)
            else:
               y_hat_norm = model.predict([test_X,extra_test_X], batch_size=batch_size, verbose=0)
            y_hat = scalar.inverse_transform(y_hat_norm)
            y_label = scalar.inverse_transform(test_Y)

            # set parameter
            sio.savemat(os.path.join(path_result, 'results.{}.{}days_smooth_{}.mat'.format(model_name, shift, it)),{'y_lab':y_label, 'y_pre':y_hat} )
