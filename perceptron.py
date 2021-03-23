import numpy as np

def getArrayFromFile(file):

    fileLines = file.readlines()
    featuresArray = np.zeros((len(fileLines),4))

    for i,line in enumerate(fileLines):
        line = line[:-9]
        instance = line.split(",")
        featuresArray[i] = instance

    return featuresArray

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
                print(str(j) + " classified wrong")

# -------------- Read in data -------------

trainFile = open("data/train.data", "r")
trainFeatures = getArrayFromFile(trainFile)
trainClasses = np.split(trainFeatures, 3)

testFile = open("data/test.data", "r")
testFeatures = getArrayFromFile(testFile)
testClasses = np.split(testFeatures, 3)

maxIter = 1

negTrainClass = np.empty([40,1])
negTrainClass.fill(-1)

posTrainClass = np.empty([40,1])
posTrainClass.fill(1)

# ----Class 1 and 2 ---------

train1 = np.hstack((trainClasses[0], negTrainClass))
train2 = np.hstack((trainClasses[1], posTrainClass))

train12 = np.concatenate((train1, train2), axis=0)
trainModel(train12, maxIter)

# ----Class 2 and 3 ---------