#include <stdio.h>
#include <math.h>

typedef String = char[20];

//linked list tree
typedef struct node{
    char* parentAttributeValue;
    char* nxtAttribute;
    struct node* branches;
} Node;

//for attributes in an instance
typedef struct attribute{
    char* name;
    char* value;
    bool continous;
    struct attribute* next;
} Attribute;

// instance typedef
typedef struct instance{
    Attribute* attribute; //array
    int attributeLength;
    char* finalResult;
} Instance

// list for all possible targets
typedef struct targetList{
    char* name;
    String* results; //array
    int length;
}TargetOptions;

// list for possible values in an attribute
typedef struct attributeList{
    char* name;
    String* Values; // array
    int length;
}AttributeList;

typedef struct attributeOptions{
    AttributeList* attValues; // array
    int length;
    bool used;
}attributeOptions;

Attribute Copy_Attribute(Attribute attributes){
    Attribute new_attribute;
    new_attribute.name = attributes.name;
    new_attribute.value = attributes.value;
    return new_attribute;
}

Instance Copy_Intstance(Instance instances){
    Instance new_instance;
    new_instance.attribute = (Attribute*)malloc(instance.attributeLength * sizeof(Attribute));
    for (int i = 0; i < instance.attributeLength; i++){
        new_instance.attribute[i] = Copy_Attribute(instances.attribute);
    }
    new_instance.finalResult = instances.finalResult;
    new_instance.attributeLength = instances.attributeLength;
    return new_instance;
}

delete_Attribute(Attribute attributes){
    free(attributes.name);
    free(attributes.value);
}

delete_Instance(Instance instances){
    for (int i = 0; i < instances.attributeLength; i++){
        delete_Attribute(instances.attribute[i]);
    }
    free(instances.attribute);
    free(instances.finalResult);
}

// Gets the entropy of a given set
double Entropy(Instance* set, int numInstances, TargetOptions TO)
{
    double entropy = 0.0;
    int targetCounts[TO.length];

    // Initialize target counts
    for (int i = 0; i < TO.length; i++){
        targetCounts[i] = 0;
    }

    // Count occurrences of each target result from the list of instances
    for (int i = 0; i < numInstances; i++){
        for (int j = 0; j < TO.length; j++){
            if (strcmp(set[i].finalResult, TO.results[j]) == 0){
                targetCounts[j]++;
                break;
            }
        }
    }

    // Calculate entropy
    for (int i = 0; i < TO.length; i++) {
        if (targetCounts[i] > 0) {
            double probability = (double)targetCounts[i] / numInstances;
            entropy += -probability * log2(probability); // sum( - p_i * log2(p_i) )
        }
    }

    return entropy;
}

// Returns sum( S_v/s * Entropy(S_v) )
// Note: this is not the same equation as in the textbook
double Information_Gain(Instance* instances, int numInstances, AttributeList attrList, TargetOptions TO){
    double result = 0.0;
    for (int v = 0; v < attrList.length; v++){
        int count = 0;
        // Create subset
        Instance subset[numInstances];

        // Count instances where the attribute and attribute value match
        for (int j = 0; j < numInstances; j++){
            for (int k = 0; k < instances[j].attributeLength; k++){
                if (strcmp(instances[j].attribute[k].name, attrList.name) == 0){
                    if (strcmp(instances[j].attribute[k].value, attrList.Values[v]) == 0){
                        subset[count] = instances[j];
                        count++;
                    }
                    break;
                }
            }
        }

        // Calculate information gain
        double probability = (double)count / numInstances;
        result += probability * Entropy(subset, count, TO);
    }
    return result;
}

char* Find_Best_Attribute(Instance* instances, int numInstances, attributeOptions AO, TargetOptions TO){
    char* bestAttr;
    double bestInfoGain = 0.0;
    bool first = true;
    int AOLength = AO.length;

    for (int i = 0; i < AOLength; i++){
        double informationGain = Information_Gain(instances, numInstances, AO[i].attributes, TO);
        if (first || informationGain < bestInfoGain) { // using minimun based on the rewritten information gain of -sum(p log2(p))
            first = false;
            bestInfoGain = informationGain;
            bestAttr = AO[i].name;
        }
    }
    return strcpy(bestAttr);
}

node Make_Tree(Instance* instances, int numInstances, attributeOptions AO, TargetOptions TO){
    
}

void Read_From_File(char* fileName){
    FILE* file = fopen(fileName, "r");
    bool isTargetLine = false;

    if (file == NULL){
        printf("File not found");
        exit();
    }
    while ((ch = fgetc(file_pointer)) != EOF) {
        printf("%c", ch);
    }
    fclose(file);
}



int main() 
{
    
    return 0;
};