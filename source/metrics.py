# import numpy as np
from __future__ import print_function
from keras import backend as K



def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    return K.mean(K.abs(y_pred - y_true) / summ * 2.0)

