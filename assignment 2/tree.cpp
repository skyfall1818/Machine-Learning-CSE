#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <string>
#include <cstring>
#include <vector>

using namespace std;

//linked list tree
// for leaf nodes: value is the target result and nxtAttribute is the count output
typedef struct node{
    char* parentAttribute;
    char* value;
    char* nxtAttribute;
    struct node* nextBranch;
    struct node* childHead;
} Node;

//for attributes in an instance
typedef struct attribute{
    string name;
    string value;
    bool continuous;
} Attribute;

// instance typedef
typedef struct instance{
    vector<Attribute> attributes;
    string finalResult;
} Instance;

// list for all possible targets
typedef struct targetList{
    string name;
    vector<string> results;
}TargetOptions;

// list for possible values in an attribute
typedef struct attributeList{
    string name;
    vector<string> values;
    bool continuous;
    bool used;
}AttributeList;

typedef struct attributeOptions{
    vector<AttributeList> attributes;
}AttributeOptions;

typedef struct continuousUsed{
    int index;
    double value;
}ContinuousBool;

Attribute Copy_Attribute(Attribute attributes){
    Attribute new_attribute;
    new_attribute.name = attributes.name;
    new_attribute.value = attributes.value;
    return new_attribute;
}

Instance Copy_Instance(Instance instance){
    Instance new_instance;
    for (int i = 0; i < instance.attributes.size(); i++){
        new_instance.attributes.push_back(Copy_Attribute(instance.attributes[i]));
    }
    new_instance.finalResult = instance.finalResult;
    return new_instance;
}

class Tree{
    private:
        AttributeOptions AO;
        TargetOptions TO;
        Node* treeHead;

        // Gets the entropy of a given set
        double Entropy(vector<Instance*> set)
        {

            double entropy = 0.0;
            int targetCounts[TO.results.size()];
            int numInstances = set.size();

            // Initialize target counts
            for (int i = 0; i < TO.results.size(); i++){
                targetCounts[i] = 0;
            }

            // Count occurrences of each target result from the list of instances
            for (int i = 0; i < numInstances; i++){
                for (int j = 0; j < TO.results.size(); j++){
                    if (set[i]->finalResult.compare(TO.results[j]) == 0){
                        targetCounts[j]++;
                        break;
                    }
                }
            }

            // Calculate entropy
            for (int i = 0; i < TO.results.size(); i++) {
                if (targetCounts[i] > 0) {
                    double probability = (double)targetCounts[i] / numInstances;
                    entropy += -probability * log2(probability); // sum( - p_i * log2(p_i) )
                }
            }
            return entropy;
        }

        // Returns sum( S_v/s * Entropy(S_v) )
        // Note: this is not the same equation as in the textbook
        double Information_Gain(vector<Instance*> instances, AttributeList attrList, int attrIndex, double conVal = 0){
            double result = 0.0;
            int numInstances = instances.size();
            if (attrList.continuous){
                vector<Instance*> subsetLower;
                vector<Instance*> subsetHigher;
                int cntLower = 0;
                int cntHigher = 0;
                for (int j = 0; j < numInstances; j++){
                    double instVal = stod(instances[j]->attributes[attrIndex].value);
                    if ( instVal > conVal){
                        subsetHigher.push_back(instances[j]);
                        cntHigher++;
                    }
                    else{
                        subsetLower.push_back(instances[j]);
                        cntLower++;
                    }
                }
                double prob = (double) cntLower / numInstances;
                result += prob * Entropy(subsetLower);
                prob = (double) cntHigher / numInstances;
                result += prob * Entropy(subsetHigher);
            }
            else{
                for (int v = 0; v < attrList.values.size(); v++){
                    int count = 0;
                    vector <Instance*> subset;
                    for (int j = 0; j < numInstances; j++){
                        if (instances[j]->attributes[attrIndex].value.compare(attrList.values[v]) == 0){
                            subset.push_back(instances[j]);
                            count++;
                        }
                    }

                    // Calculate information gain
                    double probability = (double)count / numInstances;
                    result += probability * Entropy(subset);
                }
            }
            return result;
        }

        double Gain_Ratio(vector<Instance*> instances, AttributeList attrList, int attrIndex){
            double informationGain = Information_Gain(instances, attrList, attrIndex);
            double entropy = Entropy(instances);
            return informationGain / entropy;
        }

        vector<Instance*> Merge_Sort_Instances(vector<Instance*> instances, int attrIndex){
            if (instances.size() > 1){
                vector<Instance*> sorted;
                vector<Instance*> left;
                vector<Instance*> right;
                int mid = instances.size() / 2;
                for (int i = 0; i < mid; i++){
                    left.push_back(instances[i]);
                }
                for (int i = mid; i < instances.size(); i++){
                    right.push_back(instances[i]);
                }
                left = Merge_Sort_Instances(left, attrIndex);
                right = Merge_Sort_Instances(right, attrIndex);
                int leftIndex = 0;
                int rightIndex = 0;
                for (int i = 0; i < instances.size(); i++){
                    if (leftIndex == left.size()){
                        sorted.push_back(right[rightIndex]);
                        rightIndex++;
                        continue;
                    }
                    if (rightIndex == right.size()){
                        sorted.push_back(left[leftIndex]);
                        leftIndex++;
                        continue;
                    }
                    double leftVal = stod(left[leftIndex]->attributes[attrIndex].value);
                    double rightVal = stod(right[rightIndex]->attributes[attrIndex].value);
                    if (leftVal < rightVal){
                        sorted.push_back(left[leftIndex]);
                        leftIndex++;
                    }
                    else{
                        sorted.push_back(right[rightIndex]);
                        rightIndex++;
                    }
                }
                return sorted;
            }
            else{
                return instances;
            }
        }

        int Find_Best_Attribute(vector<Instance*> instances, double* continuousVal, vector<ContinuousBool> contUsed){
            cout << "finding best attribute" << endl;
            cout << "cont Size: " << contUsed.size() << endl;
            int bestAttr;
            double bestInfoGain = 0.0;
            bool first = true;
            int AOLength = AO.attributes.size();
            double information;

            for (int a = 0; a < AOLength; a++){
                if (AO.attributes[a].used) continue; //skipping any already used attributes

                if (AO.attributes[a].continuous){
                    vector<Instance*> sortedInstances = Merge_Sort_Instances(instances, a);
                    vector<double> attrCandidates;
                    string currentTarget = "";
                    for (int i = 0; i < sortedInstances.size(); i++){
                        if (currentTarget.compare("") == 0){
                            currentTarget = sortedInstances[i]->finalResult;
                        }
                        else if (currentTarget.compare(sortedInstances[i]->finalResult) != 0){
                            double middle = stod(sortedInstances[i]->attributes[a].value) + stod(sortedInstances[i-1]->attributes[a].value);
                            middle = middle / 2;
                            bool used = false;
                            for (int c = 0; c < contUsed.size(); c++){
                                if (contUsed[c].index == a){
                                    cout << "comparing: " << contUsed[c].value << " to " << middle << endl;
                                    if( fabs(contUsed[c].value - middle) < 0.01){
                                        used = true;
                                        break;
                                    }
                                }
                            }
                            if (used) continue;
                            attrCandidates.push_back(middle);
                            currentTarget = sortedInstances[i]->finalResult;
                        }
                    }
                    bool first = true;
                    for (int i =0; i< attrCandidates.size(); i++){
                        double igVal = Information_Gain(instances, AO.attributes[a], a, attrCandidates[i]);
                        if (first){
                            *continuousVal = attrCandidates[i];
                            information = igVal;
                        }
                        else if (igVal < information) {
                            *continuousVal = attrCandidates[i];
                            information = igVal;
                        }
                    }
                }
                else{
                    information = Information_Gain(instances, AO.attributes[a], a);
                }

                if (first || information < bestInfoGain) { // using minimun based on the rewritten information gain of -sum(p log2(p))
                    first = false;
                    bestInfoGain = information;
                    bestAttr = a;
                    if (continuousVal != NULL && !AO.attributes[a].continuous){
                        *continuousVal = 0.0;
                    }
                }
            }
            return bestAttr;
        }
        
        string Get_Majority(vector<Instance*> instances){ 
            int highest = 0;
            int bestValue = 0;
            for (int i = 0; i < TO.results.size(); i++){
                int count = 0;
                for (int j = 0; j < instances.size(); j++){
                    if (instances[j]->finalResult.compare(TO.results[i]) == 0){
                        count++;
                    }
                }
                if (count > highest){
                    highest = count;
                    bestValue = i;
                }
            }
            return TO.results[bestValue];
        }

        Node* Create_leaf_Node(vector<Instance*> instances, string value){
            int cnt[TO.results.size()];
            string result = "(";
            for (int i = 0; i < TO.results.size(); i++){
                cnt [i] = 0;
            }
            for (int i = 0; i < instances.size(); i++){
                for (int r = 0; r < TO.results.size(); r++){
                    if (instances[i]->finalResult.compare(TO.results[r]) == 0){
                        cnt[r]++;
                        break;
                    }
                }
            }
            for (int i = 0; i < TO.results.size(); i++){
                result += to_string(cnt[i]);
                if (i+1 < TO.results.size()){
                    result += ",";
                }
            }
            result += ")";

            Node* leafNode = (Node*)malloc(sizeof(Node));
            leafNode->childHead = NULL;
            leafNode->nextBranch = NULL;
            leafNode->parentAttribute = NULL;

            leafNode->nxtAttribute = (char*)malloc(result.length()*sizeof(char));
            strcpy(leafNode->nxtAttribute, result.c_str());

            leafNode->value = (char*)malloc(value.length()*sizeof(char));
            strcpy(leafNode->value, value.c_str());

            return leafNode;
        }
        
        bool Check_Empty_Attribute(vector<Instance*> instances, Node* newNode, string parentMajorityTarget){
            // Check if all attributes have been used. take the most common target value
            for (int i = 0; i < AO.attributes.size(); i++){
                if (!AO.attributes[i].used){ 
                    return false;
                }
            }
            newNode->childHead = Create_leaf_Node(instances, parentMajorityTarget.c_str());
            return true;
        }

        bool Check_Same_Target(vector<Instance*> instances, Node* newNode){
            string targetTest = "";

            // Check if all instances have the same target
            for (int i = 0; i < instances.size(); i++){
                if (targetTest.compare("") == 0){
                    targetTest = instances[i]->finalResult;
                }
                else if (targetTest.compare(instances[i]->finalResult) != 0){
                    return false;
                }
            }
            newNode->childHead = Create_leaf_Node(instances, targetTest);

            return true;
        }

        Node* Make_Tree(vector<Instance*> instances, string parentAttribute, string instanceValue, string parentMajorityTarget, vector<ContinuousBool> contUsed){
            cout << "attribute: " << parentAttribute << endl;
            Node* tempNode;
            Node* nxtNode;
            double contuinuousVal;

            // initializing current node
            Node* newNode = (Node*)malloc(sizeof(Node));
                        
            if (newNode == NULL) {
                std::cerr << "Memory allocation failed" << std::endl;
                exit(1);
            }

            newNode->parentAttribute = (char*)malloc(parentAttribute.length()*sizeof(char));
            strcpy(newNode->parentAttribute, parentAttribute.c_str());

            newNode->value = (char*)malloc(instanceValue.length()*sizeof(char));
            strcpy(newNode->value, instanceValue.c_str());

            int attributeIndex = Find_Best_Attribute(instances, &contuinuousVal, contUsed);
            newNode->nxtAttribute = (char*)malloc(AO.attributes[attributeIndex].name.length()*sizeof(char));
            strcpy(newNode->nxtAttribute, AO.attributes[attributeIndex].name.c_str());
            
            newNode->childHead = NULL;
            newNode->nextBranch = NULL;

            if (instances.size() == 0){ // ran out of examples
                newNode->childHead = Create_leaf_Node(instances, parentMajorityTarget);
                return newNode;
            }

            if (Check_Empty_Attribute(instances, newNode, parentMajorityTarget) || Check_Same_Target(instances, newNode)){ // ranout of attributes or all instances have the same target
                return newNode;
            }

            string currentMajority = Get_Majority(instances);

            if (AO.attributes[attributeIndex].continuous){
                vector<Instance*> lowersubSet;
                vector<Instance*> highersubSet;
                for (int i = 0; i < instances.size(); i++){
                    if (stod(instances[i]->attributes[attributeIndex].value) > contuinuousVal){
                        highersubSet.push_back(instances[i]);
                    }
                    else{
                        lowersubSet.push_back(instances[i]);
                    }
                }
                ContinuousBool newContinuousBool;
                newContinuousBool.value = contuinuousVal;
                newContinuousBool.index = attributeIndex;
                contUsed.push_back(newContinuousBool);
                nxtNode = Make_Tree(highersubSet, AO.attributes[attributeIndex].name + ">" + to_string(contuinuousVal), "T", currentMajority, contUsed);
                nxtNode ->nextBranch = Make_Tree(lowersubSet, AO.attributes[attributeIndex].name + ">" + to_string(contuinuousVal), "F", currentMajority, contUsed);
                newNode->childHead = nxtNode;
            }
            else{
                for (int j = 0; j < AO.attributes[attributeIndex].values.size(); j++){
                    //cout << "j:" << j << endl;
                    vector<Instance*> subset;
                    for (int k = 0; k < instances.size(); k++){
                        if (instances[k]->attributes[attributeIndex].value.compare(AO.attributes[attributeIndex].values[j]) == 0){
                            subset.push_back(instances[k]);
                        }
                    }
                    AO.attributes[attributeIndex].used = true;
                    nxtNode = Make_Tree(subset, AO.attributes[attributeIndex].name,AO.attributes[attributeIndex].values[j], currentMajority, contUsed);
                    AO.attributes[attributeIndex].used = false;
                    if (newNode->childHead == NULL){
                        newNode->childHead = nxtNode;
                        tempNode = nxtNode;
                    }
                    else{
                        tempNode->nextBranch = nxtNode;
                        tempNode = nxtNode;
                    }
                }
            }
            return newNode;
        }

        void Tree_Print_Recursive(Node* node, string indent, bool first = true){
            if (!first){
                cout << indent << node->parentAttribute << " = "<< node->value;
                if (node->childHead->childHead == NULL){
                    cout << " : " << node->childHead->value << " " << node->childHead->nxtAttribute << endl;
                    return;
                }
                cout << endl;
                indent += "|   ";
            }

            Node* tempNode = node->childHead;
            while (tempNode != NULL){
                Tree_Print_Recursive(tempNode, indent, false);
                tempNode = tempNode->nextBranch;
            }
        }
        
    public:
        Tree(AttributeOptions* attrOptions, TargetOptions* targetOptions){
            AO = *attrOptions;
            TO = *targetOptions;
            treeHead = NULL;
            Print_Targets();
            Print_Attributes();
        }

        void Learn(vector<Instance> instances){
            vector<Instance*> set;
            vector<ContinuousBool> contUsed;
            for (int i = 0; i< instances.size(); i++){
                set.push_back(&instances[i]);
            }
            string majority = Get_Majority(set);
            treeHead = Make_Tree(set, "" , "", Get_Majority(set), contUsed);
        }

        void Print_Tree(){
            if (treeHead == NULL){
                return;
            }
            Tree_Print_Recursive(treeHead, "");
        }

        // These are for testing purposes
        void Print_Attributes(){
            cout << "------------Attributes------------" << endl;
            cout << "Attributes: "<<  AO.attributes.size() << endl;
            for (int i = 0; i < AO.attributes.size(); i++){
                cout << AO.attributes[i].name << ": ";
                if (AO.attributes[i].continuous){
                    cout << "continuous ";
                }
                else{
                    for (int j = 0; j < AO.attributes[i].values.size(); j++){
                        
                        cout << AO.attributes[i].values[j] << " ";
                    }
                }
                cout << endl;
            }
        }

        void Print_Targets(){
            cout << "------------Targets------------" << endl;
            cout << "Targets: "<<  TO.results.size() << endl;
            cout << TO.name << ": ";
            for (int i = 0; i < TO.results.size(); i++){
                cout << TO.results[i] << " ";
            }
            cout << endl;
        }
};


void Read_Attributes_File(char* fileName, AttributeOptions& AO, TargetOptions& TO){
    ifstream file;
    string line;
    string word;
    bool isTargetLine = false;
    bool firstWord = true;

    file.open(fileName);
    if (!file.is_open()){
        cout << "File ("<< fileName <<") Not Found" << endl;
        exit(-1);
    }
    while (getline(file, line)){
        AttributeList attrList;
        attrList.continuous = false;
        attrList.used = false;

        stringstream ss(line);
        firstWord = true;
        while (ss >> word){
            if (isTargetLine){
                if (firstWord){
                    firstWord = false;
                    TO.name = word;
                }
                else{
                    TO.results.push_back(word);
                }
            }
            else if (firstWord){
                firstWord = false;
                attrList.name = word;
            }
            else{
                if (word.compare("continuous") == 0){
                    attrList.continuous = true;
                }
                else{
                    attrList.values.push_back(word);
                }
            }
        }
        if (!isTargetLine && !firstWord){
            AO.attributes.push_back(attrList);
        }
        if (firstWord) { //check for empty line
            isTargetLine = true;
        }
    }

    file.close();
}

vector<instance> Read_Instance_File(char* fileName, AttributeOptions AO, TargetOptions TO){
    ifstream file;
    string line;
    string word;
    int count = 0;
    vector<Instance> instances;

    file.open(fileName);
    if (!file.is_open()){
        cout << "File ("<< fileName <<") Not Found" << endl;
        exit(-1);
    }
    
    while (getline(file, line)){
        stringstream ss(line);
        string word;
        Instance inst;
        count = 0;
        while (ss >> word){
            Attribute attr;

            if (count == AO.attributes.size()){
                inst.finalResult = word;
            }
            else{
                attr.name = AO.attributes[count].name;
                attr.value = word;
                inst.attributes.push_back(attr);
            }
            count++;
        }
        instances.push_back(inst);
    }

    file.close();
    return instances;
}

//This is for testing purposes
void Print_Instances(vector<Instance> instances);

int main(int argc, char *argv[]) {
    std::cout << "Number of arguments: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
    std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    AttributeOptions AO;
    TargetOptions TO;
    char* attrFile = argv[1];
    char* trainFile = argv[2];
    //string testFile = argv[3];
    Read_Attributes_File(attrFile, AO, TO);

    vector<instance> trainInstances = Read_Instance_File(trainFile, AO, TO);
    Print_Instances(trainInstances);
    
    Tree tree(&AO, &TO);
    tree.Learn(trainInstances);
    tree.Print_Tree();

    return 0;
}

//This is for testing purposes
void Print_Instances(vector<Instance> instances){
    for (int i = 0; i < instances.size(); i++){
        for (int j = 0; j < instances[i].attributes.size(); j++){
            cout << "| "<<instances[i].attributes[j].name << ": " << instances[i].attributes[j].value << " ";
        }
        cout << "| Target: " << instances[i].finalResult << endl;
    }
}