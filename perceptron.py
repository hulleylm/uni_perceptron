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

def formatData(neg, pos):
    negClass = np.hstack((neg, createColumn(len(neg),-1)))
    posClass = np.hstack((pos, createColumn(len(pos),1)))
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
    print("Train accuracy: " + str(round(accuracy, 2)) + "%")

    return bias, weights

def testBinaryModel(test, bias, weights):

    np.random.shuffle(test)
    wrong = 0

    for j,obj in enumerate(test):

        features = obj[:-1]
        actualClass = obj[4]

        activationScore = np.dot(weights, features) + bias

        if ((activationScore*actualClass) <= 0.0):
            wrong = wrong + 1
    
    accuracy = 100 - (wrong/len(test)*100)
    print("Test accuracy: " + str(round(accuracy, 2)) + "%")

def testMultiModel(bias, weightsArr, testClasses):
    
    test = np.concatenate((testClasses[0], testClasses[1], testClasses[2]), axis=0)
    right = 0

    for i,instance in enumerate(test):
        confidences = np.zeros((3))
        for j in range(len(bias)):
            activationScore = np.dot(weightsArr[j], instance) + bias[j]
            confidences[j] = activationScore
        prediction = np.argmax(confidences)
        actual = int(str(i/10)[:1])
        if (actual == prediction):
            right = right + 1
    
    accuracy = right/len(test)*100
    print("\nMulti-class test accuracy: " + str(round(accuracy, 2)) + "%")

# ---- Read in Data -----

trainClasses = readInData("train")
testClasses = readInData("test")

maxIter = 20

print("Question 3")

# ---- Question 3: Class 1 & 2 -----

print("\nClass 1 and 2")

train = formatData(trainClasses[0], trainClasses[1])
test = formatData(testClasses[0], testClasses[1])

bias, weights = trainModel(train, maxIter)
testBinaryModel(test, bias, weights)

# ---- Question 3: Class 2 & 3 -----

print("\nClass 2 and 3")

train = formatData(trainClasses[1], trainClasses[2])
test = formatData(testClasses[1], testClasses[2])

bias, weights = trainModel(train, maxIter)
testBinaryModel(test, bias, weights)

# ---- Question 3: Class 1 & 3 -----

print("\nClass 1 and 3")

train = formatData(trainClasses[0], trainClasses[2])
test = formatData(testClasses[0], testClasses[2])

bias, weights = trainModel(train, maxIter)
testBinaryModel(test, bias, weights)

# ---- Question 4: multi-class classification -----

print("\nQuestion 4\n")

#Train the individual models
bias = np.zeros((3))
weightsArr = np.zeros((3,4))

print("Model for class 1")
classes23 = np.concatenate((trainClasses[1], trainClasses[2]), axis=0)
train1 = formatData(classes23, trainClasses[0])
bias[0], weightsArr[0] = trainModel(train1, maxIter)

print("\nModel for class 2")
classes13 = np.concatenate((trainClasses[0], trainClasses[2]), axis=0)
train2 = formatData(classes13, trainClasses[1])
bias[1], weightsArr[1] = trainModel(train2, maxIter)

print("\nModel for class 3")
classes12 = np.concatenate((trainClasses[0], trainClasses[1]), axis=0)
train3 = formatData(classes12, trainClasses[2])
bias[2], weightsArr[2] = trainModel(train3, maxIter)

testMultiModel(bias, weightsArr, testClasses)

# ---- Question 5: l2 regression -----

