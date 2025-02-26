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

double PERCENTAGE_VALIDATION = 0.3;
double NOISE_ADDER = 0.02;
double MAX_NOISE = 0.2;

//linked list tree
// for leaf nodes: value is the target result and nxtAttribute is the count output
typedef struct node{
    char* parentAttribute;
    char* value;
    char* nxtAttribute;
    char* targetValue; //used for leaf node
    int* targetCounts;
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

typedef struct rule{
    vector<attribute> attributes;
    string result;
    int* targetCounts;
    double accuracy;
}Rule;

class RulePostPrune{
    private:
        AttributeOptions AO;
        TargetOptions TO;
        vector<Rule> ruleList;

        double Get_Accuracy(Rule rule, vector<Instance> testSet){
            int correct = 0;
            int count = 0;
            for (int t = 0; t < testSet.size(); t++){
                Instance instance = testSet[t];
                bool same = true;
                for (int a = 0; a < rule.attributes.size(); a++){
                    attribute attrR = rule.attributes[a];
                    if (attrR.continuous){
                        string attrName = attrR.name.substr(0, attrR.name.find('>'));
                        string floatStr = attrR.name;
                        floatStr = floatStr.substr(floatStr.find('>') + 1, floatStr.length());
                        double attrVal = atof(floatStr.c_str());
                        for (int i = 0; i < instance.attributes.size(); i++){
                            attribute attrI = instance.attributes[i];
                            if (attrName.compare(attrI.name) == 0){
                                double instanceVal = atof(attrI.value.c_str());
                                if (instanceVal <= attrVal && attrR.value.compare("T") == 0 || instanceVal > attrVal && attrR.value.compare("F") == 0){
                                    same = false;
                                }
                                break;
                            }
                        }
                    }
                    else {
                        for (int i = 0; i < instance.attributes.size(); i++){
                            attribute attrI = instance.attributes[i];
                            if (attrR.name.compare(attrI.name) == 0){
                                if (attrR.value.compare(attrI.value) != 0){
                                    same = false;
                                }
                                break;
                            }
                        }
                    }
                    if (!same){
                        break;
                    }
                }
                if (same){
                    if (rule.result.compare(instance.finalResult) == 0){
                        correct++;
                    }
                    count++;
                }
            }
            return (double)correct / count;
        }

        int* Get_Target_Counts(Rule rule, vector<Instance> testSet){
            int* targetCounts = (int*)malloc(TO.results.size()*sizeof(int));
            for (int i = 0; i < TO.results.size(); i++){
                targetCounts[i] = 0;
            }
            for (int t = 0; t < testSet.size(); t++){
                Instance instance = testSet[t];
                bool same = true;
                for (int a = 0; a < rule.attributes.size(); a++){
                    attribute attrR = rule.attributes[a];
                    if (attrR.continuous){
                        string attrName = attrR.name.substr(0, attrR.name.find('>'));
                        string floatStr = attrR.name;
                        floatStr = floatStr.substr(floatStr.find('>') + 1, floatStr.length());
                        double attrVal = atof(floatStr.c_str());
                        for (int i = 0; i < instance.attributes.size(); i++){
                            attribute attrI = instance.attributes[i];
                            if (attrName.compare(attrI.name) == 0){
                                double instanceVal = atof(attrI.value.c_str());
                                if (instanceVal <= attrVal && attrR.value.compare("T") == 0 || instanceVal > attrVal && attrR.value.compare("F") == 0){
                                    same = false;
                                }
                                break;
                            }
                        }
                    }
                    else {
                        for (int i = 0; i < instance.attributes.size(); i++){
                            attribute attrI = instance.attributes[i];
                            if (attrR.name.compare(attrI.name) == 0){
                                if (attrR.value.compare(attrI.value) != 0){
                                    same = false;
                                }
                                break;
                            }
                        }
                    }
                    if (!same){
                        break;
                    }
                }
                if (same && rule.result.compare(instance.finalResult) == 0){
                    for (int i = 0; i < TO.results.size(); i++){
                        if (TO.results[i].compare(instance.finalResult) == 0){
                            targetCounts[i] += 1;
                        }
                    }
                }
            }
            return targetCounts;
        }

        vector<Rule> Sort_Rules(vector<Rule> rules){
            if (rules.size() <= 1){
                return rules;
            }
            vector<Rule> left, right;
            int half = (double)rules.size() / 2.0;
            for (int i = 0; i < half; i++){
                left.push_back(rules[i]);
            }
            for (int i = half; i < rules.size(); i++){
                right.push_back(rules[i]);
            }
            left = Sort_Rules(left);
            right = Sort_Rules(right);

            int leftIndex = 0, rightIndex = 0;
            vector<Rule> result;
            while (leftIndex < left.size() && rightIndex < right.size()){
                if (left[leftIndex].accuracy > right[rightIndex].accuracy){
                    result.push_back(left[leftIndex]);
                    leftIndex++;
                }
                else{
                    result.push_back(right[rightIndex]);
                    rightIndex++;
                }
            }
            while (leftIndex < left.size()){
                result.push_back(left[leftIndex]);
                leftIndex++;
            }
            while (rightIndex < right.size()){
                result.push_back(right[rightIndex]);
                rightIndex++;
            }
            return result;
        }

    public:
        RulePostPrune(vector<Rule> rules, vector<Instance> testSet, AttributeOptions* attrOptions, TargetOptions* targetOptions){
            ruleList = rules;
            AO = *attrOptions;
            TO = *targetOptions;
            for (int i = 0; i < ruleList.size(); i++){
                ruleList[i].accuracy = Get_Accuracy(ruleList[i], testSet);
            }
        }

        void Prune_Rules(vector<Instance> trainSet, vector<Instance> validationSet){
            for (int i = 0; i < ruleList.size(); i++){
                bool change = true;
                while (ruleList[i].attributes.size() > 0 && change){
                    double currentAccuracy = Get_Accuracy(ruleList[i], validationSet);
                    attribute attrSave = ruleList[i].attributes[ruleList[i].attributes.size()-1];
                    ruleList[i].attributes.pop_back();
                    double newAccuracy = Get_Accuracy(ruleList[i], validationSet);
                    if (newAccuracy < currentAccuracy){
                        ruleList[i].attributes.push_back(attrSave);
                        change = false;
                        ruleList[i].accuracy = currentAccuracy;
                    }
                    else{
                        ruleList[i].accuracy = newAccuracy;
                    }
                }
                ruleList[i].targetCounts = Get_Target_Counts(ruleList[i], trainSet);
            }
            ruleList = Sort_Rules(ruleList);
        }

        double Get_Total_Accuracy(vector<Instance> testSet){
            int correct = 0;
            for (int t = 0; t< testSet.size(); t++){
                Instance instance = testSet[t];
                for (int r = 0; r < ruleList.size(); r++){
                    Rule rule = ruleList[r];
                    bool same = true;
                    for (int a = 0; a < rule.attributes.size(); a++){
                        attribute attrR = rule.attributes[a];
                        if (attrR.continuous){
                            string attrName = attrR.name.substr(0, attrR.name.find('>'));
                            string floatStr = attrR.name;
                            floatStr = floatStr.substr(floatStr.find('>') + 1, floatStr.length());
                            double attrVal = atof(floatStr.c_str());
                            for (int i = 0; i< instance.attributes.size(); i++){
                                attribute attrI = instance.attributes[i];
                                if (attrName.compare(attrI.name) == 0){
                                    double instanceVal = atof(attrI.value.c_str());
                                    if (instanceVal <= attrVal && attrR.value.compare("T") == 0 || instanceVal > attrVal && attrR.value.compare("F") == 0){
                                        same = false;
                                    }
                                    break;
                                }
                            }
                        }
                        else {
                            for (int i = 0; i< instance.attributes.size(); i++){
                                attribute attrI = instance.attributes[i];
                                if (attrR.name.compare(attrI.name) == 0){    
                                    if (attrR.value.compare(attrI.value) != 0){
                                        same = false;
                                    }
                                    break;
                                }
                            }
                        }
                        if (!same){
                            break;
                        }
                    }
                    if (same){
                        if (rule.result.compare(instance.finalResult) == 0){
                            correct++;
                        }
                        break;
                    }
                }
            }
            return (double)correct / testSet.size();
        }

        void Print_Rules(){
            for (int r = 0; r <ruleList.size(); r++){
                Rule rule = ruleList[r];
                int count = 0;
                for (int a = 0; a < rule.attributes.size(); a++){
                    attribute attrR = rule.attributes[a];
                    count++;
                    cout << attrR.name << " = " << attrR.value;
                    if (count < rule.attributes.size()){
                        cout << ", ";
                    }
                }
                string trgCnt = "(";
                for (int i = 0; i < TO.results.size(); i++){
                    ostringstream oss;
                    oss << rule.targetCounts[i];
                    trgCnt += oss.str();
                    if (i < TO.results.size() - 1){
                        trgCnt += ",";
                    }
                }
                trgCnt += ")";
                cout << " => "<<rule.result << " " << trgCnt << endl;
            }
        }
};

class Tree{
    private:
        AttributeOptions AO;
        TargetOptions TO;
        Node* treeHead;
        vector<Rule> ruleList;

        //converts double to string
        // reomves trailing zero
        string D_to_S (double value){
            ostringstream oss;
            oss << value;
            string str = oss.str();
            while (str.length() > 1 && str.find('.') != string::npos && str[str.length()-1] == '0'){
                str = str.substr(0, str.length()-1);
            }
            return str;
        }

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
                    entropy += -1 * probability * log2(probability); // sum( - p_i * log2(p_i) )
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
                    double instVal = atof(instances[j]->attributes[attrIndex].value.c_str());
                    if ( instVal > conVal){
                        subsetHigher.push_back(instances[j]);
                        cntHigher++;
                    }
                    else{
                        subsetLower.push_back(instances[j]);
                        cntLower++;
                    }
                }
                double prob = (double)cntLower / numInstances;
                result += prob * Entropy(subsetLower);
                prob = (double)cntHigher / numInstances;
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
                    double leftVal = atof(left[leftIndex]->attributes[attrIndex].value.c_str());
                    double rightVal = atof(right[rightIndex]->attributes[attrIndex].value.c_str());
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
            int bestAttr = -1;
            double bestInfoGain = -1.0;
            bool first = true;
            int AOLength = AO.attributes.size();
            double information = -1.0;
            double bestContVal = -1.0;

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
                            double middle = atof(sortedInstances[i]->attributes[a].value.c_str()) + atof(sortedInstances[i-1]->attributes[a].value.c_str());
                            middle = middle / 2;
                            bool used = false;
                            for (int c = 0; c < contUsed.size(); c++){
                                if (contUsed[c].index == a){
                                    if( fabs(contUsed[c].value - middle) < 0.01){
                                        used = true;
                                        break;
                                    }
                                }
                            }
                            if (used) continue;
                            if (attrCandidates.size() == 0 || attrCandidates[attrCandidates.size() - 1] != middle){
                                attrCandidates.push_back(middle);
                            }
                            currentTarget = sortedInstances[i]->finalResult;
                        }
                    }
                    bool f = true;
                    for (int i =0; i< attrCandidates.size(); i++){
                        double igVal = Information_Gain(instances, AO.attributes[a], a, attrCandidates[i]);
                        if (f || igVal < information){
                            f = false;
                            bestContVal = attrCandidates[i];
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
                    if (AO.attributes[a].continuous){
                        *continuousVal = bestContVal;
                    }
                    bestAttr = a;
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

        int* Get_Target_Counts(vector<Instance*> instances){
            int* cnt = (int*)malloc(TO.results.size() * sizeof(int));
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
            return cnt;
        }
        
        bool Check_Empty_Attribute(vector<Instance*> instances){
            // Check if all attributes have been used. take the most common target value
            for (int i = 0; i < AO.attributes.size(); i++){
                if (!AO.attributes[i].used){ 
                    return false;
                }
            }
            return true;
        }

        bool Check_Same_Target(vector<Instance*> instances){
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
            return true;
        }

        Node* Make_Tree(vector<Instance*> instances, string parentAttribute, string instanceValue, string parentMajorityTarget, vector<ContinuousBool> contUsed){
            Node* tempNode;
            Node* nxtNode;
            double continuousVal = 0.0;

            // initializing current node
            Node* newNode = (Node*)malloc(sizeof(Node));
                        
            if (newNode == NULL) {
                cout << "Memory allocation failed" << endl;
                exit(1);
            }

            newNode->parentAttribute = (char*)malloc(parentAttribute.length()*sizeof(char));
            strcpy(newNode->parentAttribute, parentAttribute.c_str());

            newNode->value = (char*)malloc(instanceValue.length()*sizeof(char));
            strcpy(newNode->value, instanceValue.c_str());
            
            newNode->targetCounts = Get_Target_Counts(instances);
            newNode->childHead = NULL;
            newNode->nextBranch = NULL;

            int attributeIndex = Find_Best_Attribute(instances, &continuousVal, contUsed);
            if (continuousVal == -1.0){ // end case: ran out of attributes for continuous
                newNode->targetValue = (char*)malloc(parentMajorityTarget.length()*sizeof(char));
                strcpy(newNode->targetValue, parentMajorityTarget.c_str());
                return newNode;
            }
            newNode->nxtAttribute = (char*)malloc(AO.attributes[attributeIndex].name.length()*sizeof(char));
            strcpy(newNode->nxtAttribute, AO.attributes[attributeIndex].name.c_str());

            if (instances.size() == 0){ //end case: out of instances
                newNode->targetValue = (char*)malloc(parentMajorityTarget.length()*sizeof(char));
                strcpy(newNode->targetValue, parentMajorityTarget.c_str());
                return newNode;
            }

            string currentMajority = Get_Majority(instances);
            newNode->targetValue = (char*)malloc(currentMajority.length()*sizeof(char));
            strcpy(newNode->targetValue, currentMajority.c_str());

            if (Check_Same_Target(instances) || Check_Empty_Attribute(instances)){ // end case: all instances have the same target or all attributes have been used
                return newNode;
            }

            if (AO.attributes[attributeIndex].continuous){
                vector<Instance*> lowersubSet;
                vector<Instance*> highersubSet;
                for (int i = 0; i < instances.size(); i++){
                    if (atof(instances[i]->attributes[attributeIndex].value.c_str()) > continuousVal){
                        highersubSet.push_back(instances[i]);
                    }
                    else{
                        lowersubSet.push_back(instances[i]);
                    }
                }
                ContinuousBool newContinuousBool;
                newContinuousBool.value = continuousVal;
                newContinuousBool.index = attributeIndex;
                contUsed.push_back(newContinuousBool);
                nxtNode = Make_Tree(highersubSet, AO.attributes[attributeIndex].name + ">" + D_to_S(continuousVal), "T", currentMajority, contUsed);
                nxtNode ->nextBranch = Make_Tree(lowersubSet, AO.attributes[attributeIndex].name + ">" + D_to_S(continuousVal), "F", currentMajority, contUsed);
                newNode->childHead = nxtNode;
            }
            else{
                for (int j = 0; j < AO.attributes[attributeIndex].values.size(); j++){
                    vector<Instance*> subset;
                    for (int k = 0; k < instances.size(); k++){
                        if (instances[k]->attributes[attributeIndex].value.compare(AO.attributes[attributeIndex].values[j]) == 0){
                            subset.push_back(instances[k]);
                        }
                    }
                    AO.attributes[attributeIndex].used = true;
                    nxtNode = Make_Tree(subset, AO.attributes[attributeIndex].name, AO.attributes[attributeIndex].values[j], currentMajority, contUsed);
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
                if (node->childHead == NULL){ // leafNode
                    string trgCnt = "(";
                    for (int i = 0; i < TO.results.size(); i++){
                        ostringstream str1;
                        str1 << node->targetCounts[i];
                        trgCnt += str1.str();
                        if (i < TO.results.size() - 1){
                            trgCnt += ",";
                        }
                    }
                    trgCnt += ")";
                    cout << " : " << node->targetValue << " " << trgCnt << endl;
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

        void Get_Rule_Recursion(Node* head, vector<attribute> attributes, bool first = true){
            if (!first){
                attribute att;
                att.name = head->parentAttribute;
                att.continuous = false;
                if ((int)att.name.find('>') > -1){
                    att.continuous = true;
                }
                att.value = head->value;
                attributes.push_back(att);
            }
            if (head->childHead == NULL){
                Rule rule;
                rule.attributes = attributes;
                rule.result = head->targetValue;
                rule.targetCounts = (int*)malloc(TO.results.size()*sizeof(int));
                for (int i = 0; i < TO.results.size(); i++){
                    rule.targetCounts[i] = head->targetCounts[i];
                }
                rule.accuracy = 0.0;
                ruleList.push_back(rule);
                attributes.pop_back();
                return;
            }
            Node* tempNode = head->childHead;
            while (tempNode != NULL){
                Get_Rule_Recursion(tempNode, attributes, false);
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

        void Print_Targets(){
            cout << "------------Targets------------" << endl;
            cout << "Targets: "<<  TO.results.size() << endl;
            cout << TO.name << ": ";
            for (int i = 0; i < TO.results.size(); i++){
                cout << TO.results[i] << " ";
            }
            cout << endl;
        }

        string Get_Prediction(Instance instance){
            node* currentNode = treeHead;
            while (currentNode!= NULL){
                int attributeIndex;
                for (int i = 0; i < AO.attributes.size(); i++){
                    if (strcmp(AO.attributes[i].name.c_str(), currentNode->nxtAttribute) == 0){
                        attributeIndex = i;
                        break;
                    }
                }
                currentNode = currentNode->childHead;
                if (currentNode == NULL){
                    break;
                }
                if (AO.attributes[attributeIndex].continuous){
                    double instanceVal = atof(instance.attributes[attributeIndex].value.c_str());
                    string floatStr = currentNode->parentAttribute;
                    floatStr = floatStr.substr(floatStr.find('>') + 1, floatStr.length());
                    double attrVal = atof(floatStr.c_str());
                    // short-hand since we know the current branch is binary
                    if (instanceVal <= attrVal){
                        currentNode = currentNode->nextBranch;
                    }
                }
                else{
                    while(currentNode != NULL){
                        if (strcmp(currentNode->value, instance.attributes[attributeIndex].value.c_str()) == 0){
                            break;
                        }
                        currentNode = currentNode->nextBranch;
                    }
                    if (currentNode == NULL){
                        cout << "Error Prediction" << endl;
                        exit(-1);
                    }
                }
                if (currentNode->childHead == NULL){
                    break;
                }
            }
            if (currentNode == NULL){
                cout << "Error End Prediction" << endl;
                exit(-1);
            }
            return currentNode->targetValue;
        }
        double Get_Accuracy(vector<Instance> testSet){
            int correct = 0;
            for (int i = 0; i < testSet.size(); i++){
                Instance testInstance = testSet[i];
                if (Get_Prediction(testInstance).compare(testInstance.finalResult) == 0){
                    correct++;
                }
            }
            return (double)correct / testSet.size();
        }

        bool Prune_Tree(vector<Instance> validationSet){
            double noPruneAccuracy = Get_Accuracy(validationSet);
            bool anyChange = false;
            bool update = true;
            while (update){
                vector <Node*> nodesToPrune;
                vector <Node*> nodeQueue;

                // finding all the nodes that can be pruned
                nodeQueue.push_back(treeHead);
                while(!nodeQueue.empty()){
                    Node* headNode = nodeQueue.front();
                    Node* currentNode = headNode->childHead;
                    nodeQueue.erase(nodeQueue.begin());
                    if (currentNode == NULL){
                        cout << "Error Pruning: NULL node traversal" << endl;
                        exit(-1);
                    }
                    bool allLeaf = true;
                    while (currentNode != NULL){
                        if (currentNode->childHead != NULL){
                            allLeaf = false;
                            nodeQueue.push_back(currentNode);
                        }
                        currentNode = currentNode->nextBranch;
                    }
                    if (allLeaf){
                        nodesToPrune.push_back(headNode);
                    }
                }

                // finding the best accuracy for each possible prune target
                double bestAccuracy = 0;
                Node* bestPruneNode;
                for (int n = 0; n < nodesToPrune.size(); n++){
                    Node* pruneTarget = nodesToPrune[n];
                    Node* saveChild = pruneTarget->childHead;
                    pruneTarget->childHead = NULL;
                    double accuracy = Get_Accuracy(validationSet);
                    if (accuracy > bestAccuracy){
                        bestAccuracy = accuracy;
                        bestPruneNode = pruneTarget;
                    }
                    // returning the values
                    pruneTarget->childHead = saveChild;
                }

                // updating the tree with the best prune
                if (bestAccuracy > noPruneAccuracy){
                    Node* tempNext;
                    Node* tempChild = bestPruneNode->childHead;              
                    bestPruneNode->childHead = NULL;
                    while (tempChild != NULL)
                    {
                        tempNext = tempChild->nextBranch;
                        free(tempChild->parentAttribute);
                        free(tempChild->value);
                        free(tempChild->nxtAttribute);
                        free(tempChild->targetValue);
                        free(tempChild->targetCounts);
                        free(tempChild);
                        tempChild = tempNext;
                    }
                    anyChange = true;
                }
                else{
                    update = false;
                }
            }
            return anyChange;
        }

        vector<Rule> Convert_To_Rules(){
            vector<attribute> attributes;
            Get_Rule_Recursion(treeHead, attributes);
            return ruleList;
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
}; // END Tree

vector<instance> Add_Noise(vector<Instance> instances, bool* usedList, TargetOptions TO){
    int addnoise = instances.size() * NOISE_ADDER;
    int numInstances = instances.size();
    for (int i = 0; i < addnoise; i++){
        int index = rand() % numInstances;
        while (usedList[index]){
            index = rand() % numInstances;
        }
        usedList[index] = true;

        int targetIndex = rand() % TO.results.size();
        while (instances[index].finalResult.compare(TO.results[targetIndex]) == 0){
            targetIndex = rand() % TO.results.size();
        }
        instances[index].finalResult = TO.results[targetIndex];
    }
    return instances;
}


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

//pass by values for training set and verification set
void Get_Training_and_Verification_Sets(vector<Instance> instances, vector<Instance>& trainingSet, vector<Instance>& verificationSet){
    int trainingSetSize = instances.size() * (1.0 - PERCENTAGE_VALIDATION);
    for (int i = 0; i < trainingSetSize; i++){
        trainingSet.push_back(instances[i]);
    }
    for (int i = trainingSetSize; i < instances.size(); i++){
        verificationSet.push_back(instances[i]);
    }
}


//This is for testing purposes
void Print_Instances(vector<Instance> instances);

int main(int argc, char *argv[]) {
    if (argc < 4){
        cout << "Error: need more arguments" << endl;
        exit(-1);
    }
    srand(14);
    AttributeOptions AO;
    TargetOptions TO;
    vector<Instance> trainSet, verificationSet, trainInstances;
    bool prune = false;
    bool rule = false;
    bool verbose = false;
    bool noise = false;

    bool* usedList = NULL;
    int listLength;
    double noiseLevel = 0.0;

    char* attrFile = argv[1];
    char* trainFile = argv[2];
    char* testFile = argv[3];
    for (int i = 4; i < argc; i++){
        if (strcmp(argv[i], "--prune") == 0){
            prune = true;
        }
        else if (strcmp(argv[i], "--rule") == 0){
            rule = true;
        }
        else if (strcmp(argv[i], "--v") == 0 || strcmp(argv[i], "-verbose") == 0){
            verbose = true;
        }
        else if (strcmp(argv[i], "--noise") == 0){
            noise = true;
        }
    }
    Read_Attributes_File(attrFile, AO, TO);
    vector<instance> testInstances = Read_Instance_File(testFile, AO, TO);

    if (prune || rule){
        trainInstances = Read_Instance_File(trainFile, AO, TO);
        listLength = trainInstances.size();
        Get_Training_and_Verification_Sets(trainInstances, trainSet, verificationSet);
    }
    else{
        trainSet = Read_Instance_File(trainFile, AO, TO);
        listLength = trainSet.size();
    }

    usedList = (bool*)malloc(listLength*sizeof(bool));
    for (int i = 0; i < listLength; i++){
        usedList[i] = false;
    }

    do{
        if (noise){ cout << "______________ Noise Level: " << noiseLevel << " ______________" << endl << endl; }
        Tree tree(&AO, &TO);
        tree.Learn(trainSet);
        if(verbose){
            cout << "--- Printing Decision Tree ---" << endl;
            tree.Print_Tree(); 
            cout << endl;
        }
        cout << "Accuracy of Decision Tree on training set: " << tree.Get_Accuracy(trainSet) << endl;
        cout << "Accuracy of Decision Tree on test set: " << tree.Get_Accuracy(testInstances) << endl;
        if (prune){
            if(verbose) cout << endl << "--- Pruning tree ---" << endl;
            bool update = tree.Prune_Tree(verificationSet);
            if (update){
                if(verbose) tree.Print_Tree(); 
                if(verbose) cout << endl;
                cout << "Accuracy of pruning tree on training set: " << tree.Get_Accuracy(trainSet) << endl;
                cout << "Accuracy of pruning tree on test set: " << tree.Get_Accuracy(testInstances) << endl;
            }
            else{
                if(verbose) cout << "No change" << endl;
            }
        }

        if (rule){
            if(verbose) cout << endl << "--- Rule Post Pruning ---" << endl;
            RulePostPrune rpp = RulePostPrune( tree.Convert_To_Rules(), verificationSet,&AO, &TO);
            rpp.Prune_Rules(trainSet, verificationSet);
            if(verbose) rpp.Print_Rules(); 
            if(verbose) cout << endl;
            cout << "Accuracy of rule-post pruning on training set: " << rpp.Get_Total_Accuracy(trainInstances) << endl;
            cout << "Accuracy of rule-post pruning on test set: " << rpp.Get_Total_Accuracy(testInstances) << endl;
        }

        if (noise){
            noiseLevel += NOISE_ADDER;
            if( prune || rule ){
                trainInstances = Add_Noise(trainInstances, usedList, TO);
                Get_Training_and_Verification_Sets(trainInstances, trainSet, verificationSet);
            }
            else{
                trainSet = Add_Noise(trainSet, usedList, TO);
            }
            cout << endl;
        }
    } while (noise && noiseLevel <= MAX_NOISE);
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