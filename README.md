# Textual Entailment on English Legal Text using LSTM network.
Team Project under Sabine Wehnert (PhD student, OVGU).

User is given 2 options - to Train or to Test existing models
In order to execute the program, one needs to figure out the environment variable for python executable. 

This program runs in the same format as how main.py can be executed from the command-line. 
Consider the variable for python.exe is 'python'

Then you run main.py as 'python main.py'

### Training

For training any of the 8 models, one needs to run the program in the following way:

    > python main.py
    By default, the program takes 'python' as the python launcher or the python.exe variable. If the python variable on your system is python3 or py or py3 or py2 etc., you need to go for the next option
    
        OR
    
    > {python3} main.py {python3}
    Here you provide your system defined python launcher as an extra argument. The arguments in both the curly braces will exactly be the same. Here, python3 is taken as an example. 
    
### Testing

For testing any of the 8 models, one needs to run the program in the following way:

    > python main.py
    By default, the program takes 'python' as the python launcher or the python.exe variable. If the python variable on your system is python3 or py or py3 or py2 etc., you need to go for the next option
    
        OR
    
    > {python3} main.py {python3}
    Here you provide your system defined python launcher as an extra argument. The arguments in both the curly braces will exactly be the same. Here, python3 is taken as an example. 
    
This is same as that in training. But one should take care of the model which they are interested in testing. For this, one should navigate to the model's folder and copy+paste the name of META_FILE (.meta file) of the model and then enter it once prompted on command-line.
 
     
