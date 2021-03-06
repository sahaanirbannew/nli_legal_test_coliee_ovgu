# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:32:46 2020

@author: Sachin Nandakumar
"""

'''#######################################################
                        TESTING
#######################################################'''

import os, json
import math
import sys
import numpy as np
from preprocessing import preprocessing_sabines_dataset as pre

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.utils.np_utils import to_categorical

MODEL = int(sys.argv[1])
# MODEL = 8

if MODEL == 5:
    from data_parser import data_parser_for_baseline as dp
    MODEL_NAME = 'baseline'
elif MODEL == 6:
    from data_parser import data_parser_for_POS as dp
    MODEL_NAME = 'POS'
elif MODEL == 7:
    from data_parser import data_parser_for_simneg as dp
    MODEL_NAME = 'sim_neg'
elif MODEL == 8:
    from data_parser import data_parser_for_POS_simneg as dp
    MODEL_NAME = 'POS_simneg'

'''################################################
Location of file(s) required to run the program
################################################'''

RAW_TEST_DATA = "../data/raw_data/TestData_en.xml"
LABELS_FILE = "../data/raw_data/test_labels.txt"
ROOT_MODEL_DIR = "../models/{}/attention/".format(MODEL_NAME)
META_FILE = sys.argv[2]
# META_FILE = "m_0.5849056839942932_0.5873016119003296.ckpt-280.meta"


if os.path.exists('../data/preprocessed_data/preprocessed_test_set.json'):
    PREPROCESSED_TEST_SET = "../data/preprocessed_data/preprocessed_test_set.json"            # Load json dump of test set, uncomment/comment this line
    print('\nPreprocessed test set loaded.')
else:
    PREPROCESSED_TEST_SET = pre.get_data(RAW_TEST_DATA, "TEST")     # Run preprocessing of test_set
    with open('../data/preprocessed_data/preprocessed_test_set.json', 'w') as fp:
        json.dump(PREPROCESSED_TEST_SET, fp)                        # Dump the preprocessed json file

    print('\nPreprocessing of Test Set Complete!')
    print('File {} saved to {}'.format('preprocessed_test_set.json',"../data/preprocessed_data/"))


'''################################################
Get data (premise, hypothesis, labels) for training
################################################'''

X_test = dp.get_data(PREPROCESSED_TEST_SET, "TEST")                 # Parse and get X_test data
print('\nTest set parsing complete.')

y_test = []                                                         # Read labels from text file
with open(LABELS_FILE, "r", errors='ignore') as test_labels:
    for line in test_labels:
        y_test.append(line.split(' ')[1])

y_test = to_categorical(y_test)                                     # Categorize labels into binary format


'''################################################
Restore pretrained model and calculate loss and
accuracy of input test set.
################################################'''

def plot_confusion(y_test, pred):
    labels = [0, 1]
    cm = confusion_matrix(y_test, pred, labels)
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    f1 = 2*((precision*recall)/(precision+recall))
    mcc = (cm[1][1] * cm[0][0] - cm[0][1] * cm[1][0]) / math.sqrt((cm[1][1] + cm[0][1]) * (cm[1][1]+cm[1][0]) 
                                                                  * (cm[0][0]+cm[0][1]) * (cm[0][0]+cm[1][0]))
    print('Confusion Matrix')
    print('Confusion Matrix')
    print(cm)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 Score: {}'.format(f1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap='summer')

    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.title('Confusion matrix of {}'.format(MODEL_NAME))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


tf.compat.v1.reset_default_graph()
print('\nRestoring Trained Model...')
saver = tf.compat.v1.train.import_meta_graph(ROOT_MODEL_DIR+META_FILE)
with tf.compat.v1.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(ROOT_MODEL_DIR)) # get the latest checkpoint or check the file to see which model to be restored.

    # restore all placeholders, variables & states by their tensor names for rerunning the model
    X = sess.graph.get_tensor_by_name('Placeholder:0')
    y = sess.graph.get_tensor_by_name('Placeholder_1:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    weight_decay = sess.graph.get_tensor_by_name('weight_decay:0')
    logits = sess.graph.get_tensor_by_name('output/fully_connected/BiasAdd:0')

    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(logits, name='prediction')   # applies softmax over BiRNN output to calculate predicted values

    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))                       # obtain correct predictions on comparison with actual labels
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name="accuracy")

    with tf.name_scope("loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))      # calculate loss

    # run tensorflow session to print test loss and accuracy by calling corresponding tensors for the input test set
    _, loss_test, acc_test, pred_test = sess.run([logits, loss_op, accuracy, prediction], feed_dict={X: X_test, y: y_test, keep_prob :1.0, weight_decay:0})
    print('\nTest Loss = {}, Test Accuracy = {}'.format(loss_test, acc_test))

    print('\n Test Confusion Matrix: ')
    plot_confusion(np.argmax(y_test, axis=1), np.argmax(pred_test, axis=1))
