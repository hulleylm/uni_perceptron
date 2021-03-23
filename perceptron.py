import numpy as np

def getArrayFromFile(file):
    fileLines = file.readlines()
    fileAsArray = np.zeros((len(fileLines),5))
    for i,line in enumerate(fileLines):
        line = line[:-1]
        instance = line.split(",")
        instance[4] = float(line[-1])
        fileAsArray[i] = instance
    return fileAsArray

trainFile = open("data/train.data", "r")
train = getArrayFromFile(trainFile)
