files needed
	DB/BreastCancer-attr.txt
	DB/BreastCancer-train.txt
	DB/BreastCancer-test.txt
	DB/CreditCards-attr.txt
	DB/CreditCards-train.txt
	DB/CreditCards-test.txt
	src/settings.txt
	src/settings_vary_degree.txt
	src/settings_vary_degree_CC.txt
	src/settings_vary_degree_threshold.txt
	src/settings_vary_degree_CC_threshold.txt
	src/neural_network.py
	src/neural_network_jacobi.py

note: databases are configured by running the recompile.py and the recompile_CC.py but they do not work on code01

settings file:
	settings file consists of the information for setting up magic numbers. changing these values will change how the program works
	note: I already created settings files for each test scenario
	
	Number Hidden => number of hidden nodes
	Learning Rate => learning rate
	Momentum => momentum constant
	Number Iterations => stopping criterion
	Alpha => alpha value for the Jacobi polynomial
	Beta => beta value for the Jacobi polynomial
	Degree => starting degree of Jacobi polynomial
	D Increment=1 => increment increases for the degree for each hidden node added

Compiling Code and running:
	python3 neural_network.py [settings file] [attribute file] [train file] [test file] [additional arguments]
	python3 neural_network_jacobi.py [settings file] [attribute file] [train file] [test file] [additional arguments]
	
	command options:
		-v => prints out the weights

		-t => outputs the time

		-th => threshold value. the algorithm will run until the threshold is reached. (until accuracy of network is > 0.70)

		-valid => flag to uses a validation set

		-o => outputs the current accuracies and time for every 10 iterations

	note: you can add as many argument to the end as your need

How to run programing assignment
	1) running the accuracy and time test for 250 iterations
	     i) 1:  python3 neural_network.py src/settings_vary_degree.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid
	   	2:  python3 neural_network_jacobi.py src/settings_vary_degree.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid
	     ii)1:  python3 neural_network.py src/settings_vary_degree.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid
	   	2:  python3 neural_network_jacobi.py src/settings_vary_degree.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid

	2) running the accuracy and time test for threshold
	     i) 1:  python3 neural_network.py src/settings_vary_degree_threshold.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid -th
	   	2:  python3 neural_network_jacobi.py src/settings_vary_degree_threshold.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid -th
	     ii)1:  python3 neural_network.py src/settings_vary_degree_threshold.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid -th
	   	2:  python3 neural_network_jacobi.py src/settings_vary_degree_threshold.txt DB/BreastCancer-attr.txt DB/BreastCancer-train.txt DB/BreastCancer-test.txt -t -valid -th


	