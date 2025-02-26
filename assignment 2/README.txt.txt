files needed
	tree.cpp
	tennis-attr.txt
	tennis-train.txt
	tennis-test.txt
	iris-attr.txt
	iris-train.txt
	iris-test.txt
	bool-attr.txt
	bool-train.txt
	bool-test.txt

note: All files need to be in the same directory for the program to run properly

Compiling Code:
	g++ tree.cpp -o tree

Running Code
	./tree [attribute text] [train text] [test text] [command options]
	
	command options:
		--v, --verbose => prints out the trees and rules

		--rule => adds post-rule pruning algorithm

		--noise => adds 2% noise to the training data up to 20% while recreating the trees

		--prune => adds decision tree post pruning algorithm

How to run programing assignment
	i) ./tree tennis-attr.txt tennis-train.txt tennis-test.txt --v
	ii) ./tree iris-attr.txt iris-train.txt iris-test.txt --rule
	iii) ./tree iris-attr.txt iris-train.txt iris-test.txt --rule --noise
	optional) ./tree bool-attr.txt bool-train.txt bool-test.txt --v

note: You can add "--noise" argument with the "--v" argument to have the program all the different trees

	