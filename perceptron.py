import numpy as np

def readInData(fileName):
    f = open("data/" + fileName + ".data", "r")
    features = getArrayFromFile(f)
    classesSplit = np.split(features, 3)

    return classesSplit

def getArrayFromFile(file):

    fileLines = file.readlines()
    featuresArray = np.zeros((len(fileLines),4))

    for i,line in enumerate(fileLines):
        line = line[:-9]
        instance = line.split(",")
        featuresArray[i] = instance

    return featuresArray

def getClassLabels(label):
    classArray = np.empty([40,1])
    classArray.fill(label)

    return classArray

def trainModel(train, maxIter):

    weights = np.zeros(4)
    bias = 0

    for i in range(maxIter):
        for j,obj in enumerate(train):

            features = obj[:-1]
            actualClass = obj[4]

            predictedClass = np.dot(weights, features) + bias
            print(str(j) + " predicted: " + str(predictedClass) + " actual: " + str(actualClass))

            if ((predictedClass*actualClass) <= 0.0):
                print("wrong")
                for k,weight in enumerate(weights):
                    weights[k] = weight + actualClass*features[k]
                    bias = bias + actualClass
            else:
                print("right")

    return bias, weights

# -------------- Read in data -------------

trainClasses = readInData("train")
testClasses = readInData("test")

maxIter = 3

# ----Class 1 and 2 ---------

train1 = np.hstack((trainClasses[0], getClassLabels(-1)))
train2 = np.hstack((trainClasses[1], getClassLabels(1)))

train12 = np.concatenate((train1, train2), axis=0)
bias, weights = trainModel(train12, maxIter)
testModel(bias, weights)

# ----Class 2 and 3 ---------

train2 = np.hstack((trainClasses[1], negTrainClass))
train3 = np.hstack((trainClasses[2], posTrainClass))

train12 = np.concatenate((train1, train2), axis=0)
trainModel(train12, maxIter)