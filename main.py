# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:46:59 2020

@author: Sachin Nandakumar
"""

import os, sys

def print_error_msg():
    '''
        This function is meant to print error message if wrong model number is selected
    '''
    
    print('\n Wrong Input!')
    print('Please check the options & enter appropriate option to proceed\n')
    print('Or press q to exit')
    print('-'*50)
    
def display_testing_instructions(testing_file):
    '''
        This function displays the heads-up of how the testing file should be executed
    '''
    print('Please make sure that the {} has been set with correct METAFILE PATH'.format(testing_file))
    print('If not, please visit the path of the model file and copy+paste the name of the desired METAFILE and enter it. ')
    
def print_model_options(python_launcher, mode):  
    '''
        This function prints all model combinations and forwards towards training or testing.
        
        Args: python_launcher - environment variable defined for python on system. 
              This enables running required python file for training & testing
              
              mode - 1 for training
                     2 for testing
    '''
    print('\nModels:')
    print('\t1. Baseline Model')
    print('\t2. POS Model')
    print('\t3. SimNeg Model')
    print('\t4. POS + SimNeg Model')
    print('\t5. Baseline + Attention Model')
    print('\t6. POS + Attention Model')
    print('\t7. SimNeg + Attention Model')
    print('\t8. POS + SimNeg + Attention Model')
    try:
        option_number = int(input("Select Model Number: "))
        if mode == 1:
            if 1<=option_number<=8:
                if 1<=option_number<=4:
                    os.system("{} src/training.py {}".format(python_launcher, option_number))
                else:
                    os.system("{} src/training_attention.py {}".format(python_launcher, option_number))
                
            else:
                print_error_msg()
                print_model_options()
        else:
            if 1<=option_number<=8:
                if 1<=option_number<=4:
                    display_testing_instructions('testing.py')
                    meta_file = input("Enter META_FILE: ")
                    os.system("{} src/testing.py {} {}".format(python_launcher, option_number, meta_file))
                else:
                    display_testing_instructions('testing_attention.py')
                    meta_file = input("Enter META_FILE: ")
                    os.system("{} src/testing_attention.py {} {}".format(python_launcher, option_number, meta_file))
                
            else:
                print_error_msg()
                print_model_options()
            
    except TypeError:
        if option_number.lower() == 'q':
            sys.exit("\nTerminating program!")
        print_error_msg()
        print_model_options()

def print_train_test_options(python_launcher):
    '''
        The function prints the models available for training and testing
        
        Args: python_launcher - environment variable defined for python on system. 
              This enables running required python file for training & testing
    '''
    
    print('\nRecognizing Textual Entailment in Law')
    print('-'*40)
    print('Do you want to train or test a model?')
    print('\t1. Train')
    print('\t2. Test')
    try:
        option_number = int(input('Enter option number: '))
        if option_number == 1:
            print_model_options(python_launcher, 1)
        elif option_number == 2:
            print_model_options(python_launcher, 2)
        else:
            print_error_msg()
            print_model_options()
    except TypeError:
        if option_number.lower() == 'q':
            sys.exit("\nTerminating program!")
        print_error_msg()
        print_model_options()

if __name__ == "__main__": 
    # python launcher - change the variable value if the python environment variable 
    # is different or you want to choose a specific version of python on your system
    try:
        if len(sys.argv) == 2:
            python_launcher = sys.argv[1]
        else:
            python_launcher = 'python'
        print_train_test_options(python_launcher)
    except KeyboardInterrupt as ex:
        print('-'*80)
        print(ex)
        sys.exit("Terminating Program...\nGoodbye!")
    except Exception:
        print('Wrong python launcher! Enter system defined environment python.exe variable for launching the python program')
        sys.exit("Terminating Program...\nGoodbye!")
    
    
