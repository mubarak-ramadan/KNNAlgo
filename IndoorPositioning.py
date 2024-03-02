import pandas
import csv

database = pandas.read_csv("database.csv", encoding="utf-8", sep=";")
training = pandas.read_csv("training.csv", encoding="utf-8", sep=";")




'''
    1) First I had to split the database into databaseRSS and databaseCord to store the RSS values and the 
        corresponding coordinates respectively.
    2) All leading negatives(minus sign) were also removed (This does not change anything I suppose)
    3) All RSS format are in format [RssAp1, RssAp2, RssAp3]
'''


'''
    To see the database and training set, please uncomment the code below 
'''
# print(database)
# print(training)



'''
    Reads a provided csv file into a pandas dataFrame and the convert the dataFrame to a list
    Lists are more easier to work with.
'''


def loadDataToList(filePath):
    dataFrame = pandas.read_csv(filePath, encoding="utf-8", sep=";")
    return dataFrame.values.tolist()


def loadData(filePath):
    dataFrame = pandas.read_csv(filePath, encoding="utf-8", sep=";")
    return dataFrame

'''
    Computes the minimum distance with a default values p=1 and ns= 3, else if otherwise stated.
    function takes in lists a and b which are in the form (rssAp1, rssAp2, rssAp3)
'''


def min_distance(a, b, p=1, ns=3):
    dim = len(a)

    distance = 0

    for i in range(dim):
        distance += pow(abs(int(float(a[i])) - int(float(b[i]))), p)

    distance = (distance**(1/p)) * 1/ns

    return distance

'''
    utility function to compute the weights. This function first computes the distances and evaluate the corresponding
    weights then append it to a list.
    it takes the rss of an unknown position (pos) and the training data set as input with a default value 
    of k=14 (no of RPs)
'''


def getWeight(pos, train_set, k=14):
    weights = []
    distances = []

    for i in range(k):
        dist = min_distance(pos, train_set[i], p=1)
        distances.append(1/dist)   # here I'm appending 1/d to the distance list

    for i in range(k):
        mean = sum(distances)
        weights.append(distances[i]/mean)

    return weights


'''
    function getResult computes the coordinate of the unknown location.
'''


def getResult(weight, xcord, ycord, k=14):
    xpos = 0
    ypos = 0
    for j in range(k):
        xpos += weight[j] * xcord[j]
        ypos += weight[j] * ycord[j]

    return round(xpos, 2), round(ypos, 2)


'''
    The functions declared previously are combined together to complete the algorithm which is below
'''


def KWNN_algo(rssPos, trainset, xcord, ycord, k=14, p=1, ns=3):
    print("*******************************************")
    print("************ KWNN Algorithm ****************")
    print("*******************************************")
    weights = getWeight(pos=rssPos, train_set=trainset)

    print("--> Calculating KWNN...")

    coordinate = getResult(weight=weights, xcord=xcord, ycord=ycord)
    print("KWNN algorithm completed")
    print(f" (X, Y) = {coordinate}")


# for i in range(3):
#     w = getWeight(pos=trainingList[i], train_set=databaseList)
#     res = getResult(weight=w, xcord=databaseCord.cordx, ycord=databaseCord.cordy)
#     print(res)



'''
    Performing the test
'''

databaseRSS = loadDataToList("databaseRss.csv")
databaseCord = loadData("databaseCord.csv")
testingRSS = loadDataToList("trainingRSS.csv")  # This is a list of three test points which can be accessed via indexing


KWNN_algo(rssPos=testingRSS[2], trainset=databaseRSS, xcord=databaseCord.cordx, ycord=databaseCord.cordy)