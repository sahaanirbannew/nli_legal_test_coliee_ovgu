# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:32:46 2020

@author: Sachin Nandakumar
"""

'''#######################################################
                        TESTING
#######################################################'''

import os, json
import numpy as np
from data_parser import data_parser_for_simneg as dp
from preprocessing import preprocessing_sabines_dataset as pre 


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

import tensorflow as tf
#from tensorflow.contrib import rnn
from keras.utils.np_utils import to_categorical

'''################################################
Location of file(s) required to run the program
################################################'''

RAW_TEST_DATA = "../data/raw_data/TestData_en.xml"
LABELS_FILE = "../data/raw_data/test_labels.txt"
ROOT_MODEL_DIR = "../models/sim_neg/"
META_FILE = "m_0.49056604504585266_0.6666666865348816.ckpt-1000.meta"


if os.path.exists('../data/preprocessed_data/preprocessed_test_set.json'):
    PREPROCESSED_TEST_SET = "../data/preprocessed_data/preprocessed_test_set.json"            # Load json dump of test set, uncomment/comment this line 
    print('\nPreprocessed test set loaded.')
else:
    PREPROCESSED_TEST_SET = pre.get_data(RAW_TEST_DATA, "TEST")     # Run preprocessing of test_set
    with open('../data/preprocessed_data/preprocessed_test_set.json', 'w') as fp:             
        json.dump(PREPROCESSED_TEST_SET, fp)                        # Dump the preprocessed json file
        
    print('\nPreprocessing of Test Set Complete!')
    print('File {} saved to {}'.format('preprocessed_test_set.json',"../data/preprocessed_data/"))

print(os.getcwd())


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


# '''################################################
# Define & initialize constants for lstm architecture
#     > constants for network architecture
#     > for network optimization
# ################################################'''

#learning_rate = 0.000001
#num_input = X_test.shape[2]             # dimension of each sentence 
#timesteps = X_test.shape[1]             # timesteps
#num_hidden = {1: 128, 2: 64}            # dictionary that defines number of neurons per layer 
#num_classes = 2                         # total number of classes
#num_layers = 1                          # desired number of LSTM layers
    


'''################################################
Define BiLSTM network architecture
################################################'''

def BiRNN(x, weights, bias, state_c, state_h):
    '''
        BiRNN: Defines the architecture of LSTM network for training
        Args: 
                x:          premise_hypothesis pair
                weights:    weights required to apply relu activation function over hidden layer and softmax activation over output layer 
                bias:       bias corresponding to the weights.
                state_c:    final cell state of the trained model
                state_h:    final hidden state of the trained model
        
        Returns:
                1. maladd() applied over last outputs with corresponding weights and bias
                2. concatenated forward and backward cell states
                3. whole rnn output
    '''
#    x = tf.unstack(x, timesteps, 1)
#    output = x   
#            
#    output = tf.nn.relu(tf.matmul(output, tf.cast(weights['w1'], tf.float32)) + bias['b1'])     # weights introduced to use relu activation
#    output = tf.unstack(output, timesteps, 0)
     
    with tf.compat.v1.variable_scope('lstm_test'):
        output = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_c, state_h) # create an LSTMStateTuple of pretrained cell & hidden states to get the pretrained model
    
    return tf.add(tf.matmul(output[-1], weights['out']), bias['out']) 



'''################################################
Restore pretrained model and calculate loss and 
accuracy of input test set.
################################################'''

def plot_confusion(y_test, pred):
    labels = [0, 1]
    cm = confusion_matrix(y_test, pred, labels)
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    print('Confusion Matrix')
    print(cm)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap='summer')
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    
    plt.title('Confusion matrix of POS_SimNeg')
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
    
    print('Done')
    
    # restore all placeholders, variables & states by their tensor names for rerunning the model
    X = sess.graph.get_tensor_by_name('Placeholder:0')
    y = sess.graph.get_tensor_by_name('Placeholder_1:0')
    w_out = sess.graph.get_tensor_by_name('w_out:0')
    b_out = sess.graph.get_tensor_by_name('b_out:0')
    fc_weights = {'out': w_out}
    fc_bias = {'out': b_out}
    state_c = sess.graph.get_tensor_by_name('output/lstm0/bidirectional_concat_c:0')
    state_h = sess.graph.get_tensor_by_name('output/lstm0/bidirectional_concat_h:0')
    
    # call BiRNN with placeholder X and pretrained weights and states
    # apply softmax over the BiRNN output
    with tf.name_scope("output"):
        logits = BiRNN(X, fc_weights, fc_bias, state_c, state_h)
        prediction = tf.nn.softmax(logits, name='prediction')
        
    # calculate mean of loss wrt BiRNN output and actual labels
    with tf.name_scope("loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    
    # determine model performance using accuracy metric.
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name="accuracy")
    
    # run tensorflow session to print test loss and accuracy by calling corresponding tensors for the input test set
    loss_test, acc_test, pred_test = sess.run([loss_op, accuracy, prediction], feed_dict={X: X_test, y: y_test})

    print('\nTest Loss = {}, Test Accuracy = {}'.format(loss_test, acc_test))
    
    print('\n Test Confusion Matrix: ')    
    plot_confusion(np.argmax(y_test, axis=1), np.argmax(pred_test, axis=1))