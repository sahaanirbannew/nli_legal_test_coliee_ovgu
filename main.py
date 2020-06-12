# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:46:59 2020

@author: Sachin Nandakumar
"""

import os, sys

# try:
#     from goodreads_quotes import Goodreads
# except ImportError:
#     import pip
#     pip.main(['install', 'goodreads_quotes'])
#     from goodreads_quotes import Goodreads


def print_error_msg():
    print('\n Wrong Input!')
    print('Please check the options & enter a number between 1 & 8 to proceed\n')
    print('Or press q to exit')
    print('-'*50)

def print_model_options():
    '''
    '''
    
    # python launcher - change the variable value if the python environment variable 
    # is different or you want to choose a specific version of python on your system
    python_launcher = 'python'
    
    print('Recognizing Textual Entailment in Law')
    print('Models to be trained:')
    print('\t1. Baseline Model')
    print('\t2. POS Model')
    print('\t3. SimNeg Model')
    print('\t4. POS + SimNeg Model')
    print('\t5. Baseline + Attention Model')
    print('\t6. POS + Attention Model')
    print('\t7. SimNeg + Attention Model')
    print('\t8. POS + SimNeg + Attention Model')
    
    try:
        option_number = int(input("Enter Model Number: "))
        if 1<=option_number<=8:
            if 1<=option_number<=4:
                # path = os.getcwd()
                os.system("{} src/training.py {}".format(python_launcher, option_number))
            else:
                os.system("{} src/training_attention.py {}".format(python_launcher, option_number))
            
        else:
            print_error_msg()
            print_model_options()
    except TypeError:
        if option_number.lower() == 'q':
            sys.exit("\nTerminating program!")
        print_error_msg()
        print_model_options()

if __name__ == "__main__": 
    print_model_options()
    sys.exit("\nGoodbye!")
    
    