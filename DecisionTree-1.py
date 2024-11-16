import numpy as np
import pandas as pd
import csv
import math
#import random

myname = "Aman-Samar-Ladkat"

col_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides','free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'alcohol', 'quality' ]
data = pd.read_csv("D:/MTECH/Sem-1/Foundations of ML/resources/wine-dataset.csv")

class DecisionTree():
    def __init__(self,data,threshold,columns,method,min_samples_split=2):
        self.threshold=threshold
        self.attributes=columns 
        self.way=method
        self.tree=self.learn(data)
        self.min_samples_split = min_samples_split
    def learn(self, training_set):
        tree = {} 
        select=sorted([(i,)+self.get_bestsplit(i,training_set) for i in range(self.attributes)])
        picked=select[0]    #minimum value
        feature=picked[0]   #picked least entropy
        res=picked[1]
        pos=picked[2]
        if(res<self.threshold):     #for single node tree
            tree['feature']=-1      #initial value
            label=[example[-1] for example in training_set]
            if(label.count(0)<label.count(1)):
                tree['result']=1
            else:
                tree['result']=0
            return tree
        tree['feature']=feature
        tree['position']=pos
        left_tree=[example for example in training_set if example[feature]<pos]
        label1=[example[-1] for example in left_tree]   #left tree child
        if(len(left_tree)<self.attributes or label1.count(0)==0 or label1.count(1)==0):
            leaf={}
            leaf['feature']=-1  #initial node
            if(label1.count(0)<label1.count(1)):
                leaf['result'] = 1
            else:
                leaf['result'] = 0
            tree['left_tree'] = leaf
        else:
            tree['left_tree']=self.learn(left_tree)
            
        right_tree=[example for example in training_set if example[feature]>pos]
        label2=[example[-1] for example in right_tree]  #right tree child
        if(len(right_tree)<self.attributes or label2.count(0)==0 or label2.count(1)==0):
            leaf={}
            leaf['feature']=-1  #initial node
            if(label2.count(0)<label2.count(1)):
                leaf['result'] = 1
            else:
                leaf['result'] = 0
            tree['right_tree'] = leaf
        else:
            tree['right_tree']=self.learn(right_tree)
        return tree
    def get_bestsplit(self,feature,training_set):   #for splitting the data
        training_set.sort()
        a0=[example[-1] for example in training_set].count(0)   #right 0
        b0=0    #left 0
        a1=[example[-1] for example in training_set].count(1)   #right 1
        b1=0    #left 1
        pos=-1  #initial position
        res=1   #minimum value
        for i in range(len(training_set)-1):    #last column for labels
            if((b0+b1)!=0 and (a0+a1)!=0):
                if(training_set[i][-1]!=training_set[i+1][-1]):
                    height1=(training_set[i][feature]+training_set[i+1][feature])/2  
                    height2=(b0+b1)/len(training_set)*self.way(b0/(b0+b1))+(a0+a1)/len(training_set)*self.way(a0/(a0+a1)) 
                    if(res>height2):
                        res=height2
                        pos=height1
            if(training_set[i][-1]==1):
                b1+=1
                a1-=1
            else:
                b0+=1
                a0-=1
        return res,pos
    def classify(self, test_instance):
        temp=self.tree
        while temp['feature']!=-1:  #baseline 0
            feature=temp['feature']
            position=temp['position']
            if(test_instance[feature]<position):
                temp=temp['left_tree']
            else:
                temp=temp['right_tree']   
        return temp['result']

def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def info_gain(data,split,target_name="class_type"):
    total_entropy = entropy(data[target_name])
    vals,counts = np.unique(data[split],return_counts=True)
    #calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split]==vals[i]).
                                dropna()[target_name])for i in range(len(vals))])
    
    #information gain
    ig = total_entropy-weighted_entropy
    return ig

def run_decision_tree():
    data=[]         #load dataset
    with open("D:/MTECH/Sem-1/Foundations of ML/resources/wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print("Number of records: %d" % (len(data)))
    for i in range(len(data)):
        label=[int(data[i][-1])]
        data[i]=[float(x) for x in data[i][:-1]]+label
    K = 10
    total=0
    f = open(myname+" result1.txt", "w")
    for j in range(K):
        #random.shuffle(data)
        accuracy=0
        training_set = [x for i, x in enumerate(data) if i % K != j]
        test_set = [x for i, x in enumerate(data) if i % K == j]    
        tree = DecisionTree(training_set,0.1,11,method=entropy)
        results = []
        for instance in test_set:
            result = tree.classify( instance[:-1] )
            results.append( result == instance[-1])
        accuracy = float(results.count(True))/float(len(results))
        total+=accuracy
        print(j+1, "accuracy: %.4f" % accuracy)       
        f.write("accuracy: %.4f\n" % accuracy)

    avg_accuracy=total
    avg_accuracy/=K
    print("Average accuracy: %.4f" % avg_accuracy)
    f.write("Average accuracy: %.4f" % avg_accuracy)
    f.close()

if __name__ == "__main__":
    run_decision_tree()
