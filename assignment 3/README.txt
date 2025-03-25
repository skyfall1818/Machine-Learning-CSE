files needed
	neural_network.py
	tennis-attr.txt
	tennis-train.txt
	tennis-test.txt
	iris-attr.txt
	iris-train.txt
	iris-test.txt
	identity-attr.txt
	identity-train.txt
	settings.txt
	settings_Identity_3nodes.txt
	settings_Identity_4nodes.txt
	settings_Iris.txt
	settings_Tennis.txt

note: All files need to be in the same directory for the program to run properly

settings file:
	settings file consists of the information for setting up magic numbers. changing these values will change how the program works
	note: I already created settings files for each test scenario
	
	Number Hidden => number of hidden nodes
	Learning Rate => learning rate
	Momentum => momentum constant
	Number Iterations => stopping criterion

Compiling Code and running:
	python3 neural_network.py [settings file] [attribute file] [train file] [test file] [additional arguments]
	
	command options:
		-v=> prints out the weights

		-n => adds noise to the data and outputs accuracies

		-noTest => for Identity since there is no test.txt

		-valid => flag to uses a validation set

	note: you can add as many argument to the end as your need (invalid arguments are ignored)

How to run programing assignment
	i) 1:  python3 neural_network.py settings_Identity_3nodes.txt identity-attr.txt identity-train.txt identity-train.txt -noTest
	   2:  python3 neural_network.py settings_Identity_4nodes.txt identity-attr.txt identity-train.txt identity-train.txt -noTest
	ii) python3 neural_network.py settings_Tennis.txt tennis-attr.txt tennis-train.txt tennis-test.txt
	iii) validation set: python3 neural_network.py settings_Iris.txt iris-attr.txt iris-train.txt iris-test.txt -n -valid
	     no validation set: python3 neural_network.py settings_Iris.txt iris-attr.txt iris-train.txt iris-test.txt -n

	