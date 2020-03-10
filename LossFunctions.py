# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:19:06 2019

@author: egarciamarcos
"""

from keras import backend as K
import tensorflow as tf

""" Dice Loss Function """
"""
Here is a dice loss for keras which is smoothed to approximate a linear (L1) loss.
It ranges from 1 to 0 (no error), and returns results similar to binary crossentropy
"""
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
#    
## Test
#y_true = np.array([[0,0,1,0],[0,0,1,0],[0,0,1.,0.]])
#y_pred = np.array([[0,0,0.9,0],[0,0,0.1,0],[1,1,0.1,1.]])
#
#r = dice_coef_loss(
#    K.theano.shared(y_true),
#    K.theano.shared(y_pred),
#).eval()
#print('dice_coef_loss',r)
#
#
#r = keras.objectives.binary_crossentropy(
#    K.theano.shared(y_true),
#    K.theano.shared(y_pred),
#).eval()
#print('binary_crossentropy',r)
#print('binary_crossentropy_scaled',r/r.max())
## TYPE                 |Almost_right |half right |all_wrong
## dice_coef_loss      [ 0.00355872    0.40298507  0.76047904]
## binary_crossentropy [ 0.0263402     0.57564635  12.53243514]


""" Discriminative Instance Loss Function"""

def discriminative_loss(delta_v=0.5, delta_d=1.5, gamma=1e-3):
    
    def discriminative_instance(y_true, y_pred):
        """Discriminative loss between an output tensor and a target tensor.
    
        Args:
            y_true: A tensor of the same shape as y_pred.
            y_pred: A tensor of the vector embedding
    
        Returns:
            tensor: Output tensor.
        """
    
        def temp_norm(ten, axis=None):
            if axis is None:
                axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(ten) - 1
            return K.sqrt(K.epsilon() + K.sum(K.square(ten), axis=axis))
    
        rank = K.ndim(y_pred)
        channel_axis = 1 if K.image_data_format() == 'channels_first' else rank - 1
        axes = [x for x in list(range(rank)) if x != channel_axis]
    
        # Compute variance loss
        cells_summed = tf.tensordot(y_true, y_pred, axes=[axes, axes])
        n_pixels = K.cast(tf.count_nonzero(y_true, axis=axes), dtype=K.floatx()) + K.epsilon()
        n_pixels_expand = K.expand_dims(n_pixels, axis=1) + K.epsilon()
        mu = tf.divide(cells_summed, n_pixels_expand)
    
        D_v = K.constant(delta_v, dtype=K.floatx())
        mu_tensor = tf.tensordot(y_true, mu, axes=[[channel_axis], [0]])
        L_var_1 = y_pred - mu_tensor
        L_var_2 = K.square(K.relu(temp_norm(L_var_1) - D_v))
        L_var_3 = tf.tensordot(L_var_2, y_true, axes=[axes, axes])
        L_var_4 = tf.divide(L_var_3, n_pixels)
        L_var = K.mean(L_var_4)
    
        # Compute distance loss
        mu_a = K.expand_dims(mu, axis=0)
        mu_b = K.expand_dims(mu, axis=1)
    
        diff_matrix = tf.subtract(mu_b, mu_a)
        L_dist_1 = temp_norm(diff_matrix)
        L_dist_2 = K.square(K.relu(K.constant(2 * delta_d, dtype=K.floatx()) - L_dist_1))
        diag = K.constant(0, dtype=K.floatx()) * tf.diag_part(L_dist_2)
        L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
        L_dist = K.mean(L_dist_3)
    
        # Compute regularization loss
        L_reg = gamma * temp_norm(mu)
        L = L_var + L_dist + K.mean(L_reg)
    
        return L
    
    return discriminative_instance