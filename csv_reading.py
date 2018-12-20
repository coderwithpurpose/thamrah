import pandas as pd
import numpy as np

import random
import math



# def splitDataset(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     print(len(dataset))
#     print(random.randrange(len(copy)))
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]



## substracting the mean of each nnun=merical column
def minus_mean(vector):
    mean = np.mean(vector)
    vector[:] = [x - mean for x in vector]
    return vector

def adjust_labels(dataset):
    w,l = np.shape(dataset)
    for i in range(0,w):
        if  '<=50K' in dataset[i,-1]:
            dataset[i,-1] = -1
        else:
            dataset[i,-1] = 1
    return dataset

## return label and features
def spliiter(dataset):
    w,l = np.shape(dataset)
    features = []
    # features = []
    labels= np.zeros(w)
    labels[:] = dataset[:,-1]
    for j in range(0,w):
        # print(dataset[j,:(l-1)])
        features.append(dataset[j, :(l-1)])
    print(np.shape(features), np.shape(labels))
    return features,labels

# estimating the accuracy of svm_sgd
def getAccuracy(a,b,features,labels):
    estFxn = features*a + b
    predictedLabels =(labels)
    for i in range(0,len(labels)):
        if estFxn[i] < 0:
            predictedLabels[i] = -1
        else:
            predictedLabels[i] = 1
    return(sum(predictedLabels == labels)/ len(labels))

# reading csv file and splitting into test and train
df = pd.read_csv('train.csv')
test_array = df.values
l,w = np.shape(test_array)

# making the variance of the numerical values to be = 1
idxs = [0,2,4,10,11,12]
for i in range(0,len(idxs)):
    # print(idxs[i])
    print('should be all numbers',test_array[:,idxs[i]])
    test_array[:,i] = minus_mean(test_array[:,idxs[i]])

# checking wehther the mean is close to 0
# print(np.mean(test_array[:,4]))
# print(np.mean(test_array[:,11]))
# print(np.mean(test_array[:,12]))
# print(np.mean(test_array[:,10]))
# print(np.mean(test_array[:,2]))
# print(np.mean(test_array[:,0]))

# now shuffle and take 90-10 % of train to test data
np.random.shuffle(test_array)
test_array = adjust_labels(test_array)
# print(test_array[:,-1])
l = int(len(test_array)*0.9)
traininset = test_array[:l,:]
X,Y = spliiter(traininset)
testset = test_array[l:,:]
Xtest, Ytest = spliiter(testset)
# print(np.shape(traininset))
# print(np.mean(traininset[:,4]))
print(np.shape(X), np.shape(Y))

# print('labels = ' , Y)
# print('first couple of fetures = ', len(X[0]))

# now we substracted by the mean and splitted the dataset into labels and fetures

# start the model
numEpochs = 100
numStepsPerEpoch = 500
nStepsPerPlot = 30
evalidationSetSize = 50
c1 = 0.01
c2 = 50

lambda_vals = {0.001, 0.01, 0.1, 1}
bestAccuracy = 0


