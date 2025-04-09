files needed
	Genetic_Algorithm.py
	tennis-attr.txt
	tennis-train.txt
	tennis-test.txt
	iris-attr.txt
	iris-train.txt
	iris-test.txt
	settings_iris_replacement.txt
	settings_iris_generation.txt
	settings_tennis_replacement.txt
	settings_tennis_generation.txt

note: All files need to be in the same directory for the program to run properly

settings file:
	settings file consists of the information for setting up magic numbers. changing these values will change how the program works
	note: I already created settings files for each test scenario
	note: not all values work with certain conditions
	
	Population Size => size of each population
	Replacement Rate => rate of replacement per generation
	Mutation Rate => rate of a mutation occurring
	Number Iterations => iterations run for the GA
	Expanding Mutation Rate => chance of a mutation where the size of an indvidual increases
	Strategy => type of strategy used for fitness selection 
		    strategies: F = fitness selection | T = tournament selection | R = Rank selection

Compiling Code and running:
	python3 neural_network.py [settings file] [attribute file] [train file] [test file] [additional arguments]
	
	additional arguments:
		-v=> prints out the rules in a human-readable format
			format:
				discrete variables: rule[#]: [attribute name] = [variables], ..., => [target value]
				continuous variables: rule[#]: [attribute name] = [value range], ..., => [target value]

		-g => uses the Number Iterations and print out the accuracies at each 10% iterations generated

		-r => replace the replacement rate with 0.1, 0.2, ..., 0.9 and outputs the accuracies

	note: you can add as many argument to the end as your need (invalid arguments are ignored) (-r and -g do not work together)

How to run programing assignment
	i)    python3 Genetic_Algorithm.py settings.txt tennis-attr.txt tennis-train.txt tennis-test.txt -v
	ii)   python3 Genetic_Algorithm.py settings.txt iris-attr.txt iris-train.txt iris-test.txt -v
	iii)  python3 Genetic_Algorithm.py settings_iris_generation.txt iris-attr.txt iris-train.txt iris-test.txt -g
	iiii) python3 Genetic_Algorithm.py settings_iris_replacement.txt iris-attr.txt iris-train.txt iris-test.txt -r

	