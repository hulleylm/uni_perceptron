import numpy as np

def readInData(fileName):
    f = open("data/" + fileName + ".data", "r")
    features = getArrayFromFile(f)
    classesSplit = np.split(features, 3)

    return classesSplit

def getArrayFromFile(file):

    fileLines = file.readlines()
    numInstances = len(fileLines)
    featuresArray = np.zeros((numInstances,4))

    for i,line in enumerate(fileLines):
        line = line[:-9]
        instance = line.split(",")
        featuresArray[i] = instance

    # featuresArray = normalise(featuresArray)

    return featuresArray

def createColumn(size, label):
    column = np.empty([size,1])
    column.fill(label)

    return column

def normalise(arr):
    maxFeatures = np.amax(arr, axis = 0)
    minFeatures = np.amin(arr, axis = 0)

    for i in range(len(arr)):
        for j in range(len(maxFeatures)):
            feature = arr[i][j]
            arr[i][j] = (feature - minFeatures[j])/(maxFeatures[j] - minFeatures[j])
    return arr

def formatData(first, second, classType):
    negClass = np.hstack((classType[first-1], createColumn(len(classType[first-1]),-1)))
    posClass = np.hstack((classType[second-1], createColumn(len(classType[second-1]),1)))
    combined = np.concatenate((negClass, posClass), axis=0)

    return combined

def trainModel(train, maxIter):

    bias = 0
    weights = np.zeros(4)
    
    for i in range(maxIter):

        np.random.shuffle(train)
        wrong = 0

        for j,obj in enumerate(train):

            features = obj[:-1]
            actualClass = obj[4]

            activationScore = np.dot(weights, features) + bias

            if ((activationScore*actualClass) <= 0.0):
                wrong = wrong + 1
                for k,weight in enumerate(weights):
                    weights[k] = weight + actualClass*features[k] #Update the weights according to the rule
                bias = bias + actualClass
    
    accuracy = 100 - (wrong/len(train)*100)
    print("Train accuracy: " + str(accuracy) + "%")

    return bias, weights

def testModel(test, bias, weights):

    np.random.shuffle(test)
    wrong = 0

    for j,obj in enumerate(test):

        features = obj[:-1]
        actualClass = obj[4]

        activationScore = np.dot(weights, features) + bias

        if ((activationScore*actualClass) <= 0.0):
            wrong = wrong + 1
    
    accuracy = 100 - (wrong/len(test)*100)
    print("Test accuracy: " + str(accuracy) + "%")

# ---- Read in Data -----

trainClasses = readInData("train")
testClasses = readInData("test")

maxIter = 20

# ---- Question 3: Class 1 & 2 -----

print("\nClass 1 and 2")

train = formatData(1, 2, trainClasses)
test = formatData(1, 2, testClasses)

bias, weights = trainModel(train, maxIter)
testModel(test, bias, weights)

# ---- Question 3: Class 2 & 3 -----

print("\nClass 2 and 3")

train = formatData(2, 3, trainClasses)
test = formatData(2, 3, testClasses)

bias, weights = trainModel(train, maxIter)
testModel(test, bias, weights)

# ---- Question 3: Class 1 & 3 -----

print("\nClass 1 and 3")

train = formatData(1, 3, trainClasses)
test = formatData(1, 3, testClasses)

bias, weights = trainModel(train, maxIter)
testModel(test, bias, weights)