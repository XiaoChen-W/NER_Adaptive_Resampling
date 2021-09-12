# -*- coding: utf-8 -*-


import tensorflow as tf
# Implementation of Focal Loss and Dice Loss using Tensorflow 1.12.0

# Apppy loss functions in Softmax Layer
def softmax_layer(logits,labels,num_labels,mask, loss_type):
    
    if loss_type not in ['Focal', 'Dice']:
        raise ValueError("Unidentified Loss Function")
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    if loss_type == 'Focal':    
        loss = focal_loss_softmax(labels, logits,gamma = 2)
    if loss_type == 'Dice':    
        loss = dice_loss(labels, logits,num_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict

# Implementation of Dice Loss using Tensorflow 1.12.0
def dice_loss(label, logits, n_classes, smooth=1.e-5):
    
    epsilon = 1.e-6
    softmax_prob = tf.nn.softmax(logits,dim = -1)
    y_pred = tf.clip_by_value(softmax_prob, epsilon, 1. - epsilon)
    y_true = tf.one_hot(label, depth=y_pred.shape[1])
    numerator = 2*tf.reduce_sum(y_true*y_pred)+smooth
    denominator = tf.reduce_sum(y_pred*y_pred)+tf.reduce_sum(y_true*y_true)+smooth
    dice_coe = tf.divide(numerator, denominator)
    return 1 - dice_coe

# Implementation of Dice Loss using Tensorflow 1.12.0
# Gamma is set to 2 according to the authors of (Lin et al. 2017) recommended
def focal_loss_softmax(labels,logits,gamma=2):

    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L
