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
    bool continous;
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
    bool continous;
    bool used;
}AttributeList;

typedef struct attributeOptions{
    vector<AttributeList> attributes;
}AttributeOptions;

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
        double Information_Gain(vector<Instance*> instances, AttributeList attrList, int attrIndex){
            double result = 0.0;
            int numInstances = instances.size();
            //cout << "value size: "<< attrList.values.size() << endl;
            for (int v = 0; v < attrList.values.size(); v++){
                int count = 0;
                // Create subset
                vector <Instance*> subset;
                //cout << "IG - Num: "<< numInstances << endl;
                // Count instances where the attribute and attribute value match
                for (int j = 0; j < numInstances; j++){
                    //cout << "j: " << j << endl;
                    if (instances[j]->attributes[attrIndex].value.compare(attrList.values[v]) == 0){
                        subset.push_back(instances[j]);
                        count++;
                    }
                }

                // Calculate information gain
                double probability = (double)count / numInstances;
                //cout << "Test" << endl;
                result += probability * Entropy(subset);
                //cout << "IGend" << endl;
            }
            //cout << "returning" << endl;
            return result;
        }

        int Find_Best_Attribute(vector<Instance*> instances){
            int bestAttr;
            double bestInfoGain = 0.0;
            bool first = true;
            int AOLength = AO.attributes.size();
            for (int i = 0; i < AOLength; i++){
                if (AO.attributes[i].used) continue; //skipping any already used attributes
                //cout << "testFA" << endl;
                double informationGain = Information_Gain(instances, AO.attributes[i], i);
                //cout << "FA2" << endl;
                if (first || informationGain < bestInfoGain) { // using minimun based on the rewritten information gain of -sum(p log2(p))
                    first = false;
                    bestInfoGain = informationGain;
                    bestAttr = i;
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

        Node* Make_Tree(vector<Instance*> instances, string parentAttribute, string instanceValue, string parentMajorityTarget){
            Node* tempNode;
            Node* nxtNode;

            // initializing current node
            Node* newNode = (Node*)malloc(sizeof(Node));

            newNode->parentAttribute = (char*)malloc(parentAttribute.length()*sizeof(char));
            strcpy(newNode->parentAttribute, parentAttribute.c_str());

            newNode->value = (char*)malloc(instanceValue.length()*sizeof(char));
            strcpy(newNode->value, instanceValue.c_str());

            int attributeIndex = Find_Best_Attribute(instances);
            newNode->nxtAttribute = (char*)malloc(AO.attributes[attributeIndex].name.length()*sizeof(char));
            strcpy(newNode->nxtAttribute, AO.attributes[attributeIndex].name.c_str());
            
            newNode->childHead = NULL;
            newNode->nextBranch = NULL;
            
            if (newNode == NULL) {
                std::cerr << "Memory allocation failed" << std::endl;
                exit(1);
            }

            if (instances.size() == 0){
                newNode->childHead = Create_leaf_Node(instances, parentMajorityTarget);
                return newNode;
            }

            if (Check_Empty_Attribute(instances, newNode, parentMajorityTarget) || Check_Same_Target(instances, newNode)){
                return newNode;
            }

            string currentMajority = Get_Majority(instances);

            for (int j = 0; j < AO.attributes[attributeIndex].values.size(); j++){
                //cout << "j:" << j << endl;
                vector<Instance*> subset;
                for (int k = 0; k < instances.size(); k++){
                    if (instances[k]->attributes[attributeIndex].value.compare(AO.attributes[attributeIndex].values[j]) == 0){
                        subset.push_back(instances[k]);
                    }
                }
                AO.attributes[attributeIndex].used = true;
                nxtNode = Make_Tree(subset, AO.attributes[attributeIndex].name,AO.attributes[attributeIndex].values[j], currentMajority);
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
        }

        void Learn(vector<Instance> instances){
            vector<Instance*> set;
            for (int i = 0; i< instances.size(); i++){
                set.push_back(&instances[i]);
            }
            treeHead = Make_Tree(set, "" , "", Get_Majority(set));
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
                for (int j = 0; j < AO.attributes[i].values.size(); j++){
                    cout << AO.attributes[i].values[j] << " ";
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


void Read_Attributes_File(string fileName, AttributeOptions& AO, TargetOptions& TO){
    ifstream file;
    string line;
    string word;
    bool isTargetLine = false;
    bool firstWord = true;

    file.open(fileName.c_str());
    if (!file.is_open()){
        cout << "File ("<< fileName <<") Not Found" << endl;
        exit(-1);
    }
    while (getline(file, line)){
        stringstream ss(line);
        firstWord = true;
        AttributeList attrList;
        attrList.continous = false;
        attrList.used = false;

        while (getline(ss, word, ' ')){
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
                    attrList.continous = true;
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

vector<instance> Read_Instance_File(string fileName, AttributeOptions AO, TargetOptions TO){
    ifstream file;
    string line;
    string word;
    int count = 0;
    vector<Instance> instances;

    file.open(fileName.c_str());
    if (!file.is_open()){
        cout << "File ("<< fileName <<") Not Found" << endl;
        exit(-1);
    }
    
    while (getline(file, line)){
        stringstream ss(line);
        Instance inst;
        count = 0;
        while (getline(ss, word, ' ')){
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

    string attrFile(argv[1]);

    AttributeOptions AO;
    TargetOptions TO;
    string trainFile = argv[2];
    //string testFile = argv[3];
    Read_Attributes_File(attrFile, AO, TO);

    vector<instance> trainInstances = Read_Instance_File(trainFile, AO, TO);

    Tree tree(&AO, &TO);
    tree.Learn(trainInstances);
    tree.Print_Tree();

    return 0;
}

//This is for testing purposes
void Print_Instances(vector<Instance> instances){
    for (int i = 0; i < instances.size(); i++){
        for (int i = 0; i < trainInstances.size(); i++){
            for (int j = 0; j < trainInstances[i].attributes.size(); j++){
                cout << "| "<<trainInstances[i].attributes[j].name << ": " << trainInstances[i].attributes[j].value << " ";
            }
            cout << "| Target: " << trainInstances[i].finalResult << endl;
        }
    }
}