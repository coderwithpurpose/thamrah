from sklearn import svm
import csv
import random
import math
import numpy as np


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

# here we splitting dataset by classes,
# returning a 2d array with trwo classes and all features
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def main():
    # overall_accuracy = []
    clf = svm.SVC()
    overall_accuracy = []
    for idx in range(0, 10):
        filename = 'pima-indians-diabetes.csv'
        splitRatio = 0.80
        dataset = loadCsv(filename)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
        # prepare model

        separate = separateByClass(trainingSet)
        # x = separate[0]
        # y = separate[1]
        # x = np.zeros(len(separate))
        x = []
        nested = []
        print(np.shape(separate))
        print(np.shape(separate[0]))
        xidx , yidx = np.shape(separate[0])
        xidx2 , yidx2 = np.shape(separate[1])
        print(np.shape((separate[1])))
        for i in range (0,xidx):
            # print(xidx)
            nested.append(separate[0][i])
        for j in range (0,xidx2):
            nested.append(separate[1][j])
        print(np.shape(nested))

        # print(np.shape(separate)
        #
        # nested = {separate[0],separate[1]}
        for i in range(0,len(separate[0])):
            x.append(1)
        for j in range(0,len(separate[1])):
            x.append(0)
        print(np.shape(x))
        clf.fit(nested,x)
        print(np.shape(testSet))
        # print(clf.predict(testSet))
        predection = clf.predict(testSet)
        print(len(predection))
        accuracy = getAccuracy(testSet,predection)
        print(accuracy)
        overall_accuracy.append(accuracy)
    print("overall accuracy over 10 iterations = " , mean(overall_accuracy))
        # test model
    #     predictions = getPredictions(summaries, testSet)
    #     accuracy = getAccuracy(testSet, predictions)
    #     print('Accuracy: {0}%').format(accuracy)
    #     overall_accuracy.append(accuracy)
    # print("accuracy over 10 trials after removing zeros = " ,mean(overall_accuracy))


main()