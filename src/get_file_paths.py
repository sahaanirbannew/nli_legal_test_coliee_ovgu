# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:59:39 2020

@author: Sachin Nandakumar
"""

def get_file_paths_for_models_without_attention(MODEL_NAME):
    PREPROCESSED_TRAIN_SET = "../data/preprocessed_data/preprocessed_training_set.json"
    PREPROCESSED_REDUCED_TRAIN_SET = "../data/preprocessed_data/preprocessed_reduced_training_set.json"
    PREPROCESSED_VALIDATION_SET = "../data/preprocessed_data/preprocessed_validation_set.json"
    SAVE_MODEL_TO = "../models/{}/".format(MODEL_NAME)
    SAVE_STATES_TO = "../states/{}/states".format(MODEL_NAME)
    SAVE_LOGS_TO = "../tensorBoardLogs/{}/".format(MODEL_NAME)
    TRAINING_LOG = "../logs/{}/training_performance_log.txt".format(MODEL_NAME)
    return PREPROCESSED_TRAIN_SET, PREPROCESSED_REDUCED_TRAIN_SET, PREPROCESSED_VALIDATION_SET, \
        SAVE_MODEL_TO, SAVE_STATES_TO, SAVE_LOGS_TO, TRAINING_LOG, MODEL_NAME

def get_file_paths_for_attentionmodels(MODEL_NAME):
    PREPROCESSED_TRAIN_SET = "../data/preprocessed_data/preprocessed_training_set.json"
    PREPROCESSED_REDUCED_TRAIN_SET = "../data/preprocessed_data/preprocessed_reduced_training_set.json"
    PREPROCESSED_VALIDATION_SET = "../data/preprocessed_data/preprocessed_validation_set.json"
    SAVE_MODEL_TO = "../models/{}/attention/".format(MODEL_NAME)
    SAVE_STATES_TO = "../states/{}/attention/states".format(MODEL_NAME)
    SAVE_SCORES_TO = "../attention_scores/{}/attention_scores".format(MODEL_NAME)
    SAVE_LOGS_TO = "../tensorBoardLogs/{}/attention/".format(MODEL_NAME)
    TRAINING_LOG = "../logs/{}/attention/training_performance_log.txt".format(MODEL_NAME)
    return PREPROCESSED_TRAIN_SET, PREPROCESSED_REDUCED_TRAIN_SET, PREPROCESSED_VALIDATION_SET, \
        SAVE_MODEL_TO, SAVE_STATES_TO, SAVE_SCORES_TO, SAVE_LOGS_TO, TRAINING_LOG, MODEL_NAME

def get_file_paths_main(model):
    if model == 1:
        return get_file_paths_for_models_without_attention('baseline')
    elif model == 2:
        return get_file_paths_for_models_without_attention('POS')
    elif model == 3:
        return get_file_paths_for_models_without_attention('sim_neg')
    elif model == 4:
        return get_file_paths_for_models_without_attention('POS_simneg')
    elif model == 5:
        return get_file_paths_for_attentionmodels('baseline')
    elif model == 6:
        return get_file_paths_for_attentionmodels('POS')
    elif model == 7:
        return get_file_paths_for_attentionmodels('sim_neg')
    elif model == 8:
        return get_file_paths_for_attentionmodels('POS_simneg')
    else:
        print('Wrong model choice!')
        