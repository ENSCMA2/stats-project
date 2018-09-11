import csv
import operator
import math
import random
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import *

for x in range(0, 150):

  #imports the data and gets it ready to be processed
  def loadDataset(file, trainingSet=[], testingSet=[]):
    csvfile = open(file, 'r+')
    lines = csv.reader(csvfile)
    data = list(lines)
    randomNumber = random.randint(1,5)
    for i in range(len(data)):
      if (i-randomNumber)%5==0:
        testingSet.append(data[i])
      else:
        trainingSet.append(data[i])

  #applying the loadDataset function to our data
  trainingSet = []
  testingSet = []
  loadDataset('tweets.csv', trainingSet, testingSet) #make sure to change the file names to whatever you named them, and make sure they are in the same directory as this file
  def prepData(training, relevantWords = [[],[],[],[]]):
    stop_words = set(stopwords.words('english'))
    urgency_words = ["urgent", "need", "require", "serious", "critical", "stress"]
    stemmer = SnowballStemmer('english')
    for tweet in training:
      for word in word_tokenize(tweet[0]):
        if word[0]=="@":
          tweet[0] = tweet[0].replace(word, "")
      for c in "!@#%&*()[]{-}/?<\">":
        tweet[0] = tweet[0].replace(c, "").lower()
      tweet[0] = [w for w in word_tokenize(tweet[0]) if not w in stop_words]
      for word in tweet[0]:
        word = stemmer.stem(word)
        if not word in urgency_words:
          if tweet[1]=="Food":
            relevantWords[0].append(word)
          elif tweet[1]=="Energy":
            relevantWords[1].append(word)
          elif tweet[1]=="Medical":
            relevantWords[2].append(word)
          elif tweet[1]=="Water":
            relevantWords[3].append(word)

  def testPrep(testing):
    stop_words = set(stopwords.words('english'))
    urgency_words = ["urgent", "need", "require", "serious", "critical", "stress"]
    stemmer = SnowballStemmer('english')
    for tweet in testing:
      for word in word_tokenize(tweet[0]):
        if word[0]=="@":
          tweet[0] = tweet[0].replace(word, "")
      for c in "!@#%&*()[]{-}/?<\">":
        tweet[0] = tweet[0].replace(c, "").lower()
      tweet[0] = [w for w in word_tokenize(tweet[0]) if not w in stop_words]
      for word in tweet[0]:
        word = stemmer.stem(word)

  def getFeatures(tweet, relevantWords, featureVector = [0,0,0,0]):
      for word in tweet[0]:
        for i in range(len(relevantWords)):
          if word in relevantWords[i]:
            featureVector[i] +=1
      return featureVector

  relevantWords = [[],[],[],[]]
  prepData(trainingSet, relevantWords)
  foodList = []
  waterList = []
  energyList = []
  medicalList = []
  trainingFeatureVectorList = []
  for tweet in trainingSet:
    featureVector = [0,0,0,0]
    trainingFeatureVectorList.append(getFeatures(tweet, relevantWords, featureVector))
    trainingFeatureVectorList[-1].append(tweet[1])
    foodList.append(featureVector[0])
    energyList.append(featureVector[1])
    medicalList.append(featureVector[2])
    waterList.append(featureVector[3])

  foodMax = max(foodList)
  foodMin = min(foodList)
  waterMax = max(waterList)
  waterMin = min(waterList)
  energyMax = max(energyList)
  energyMin = min(energyList)
  medicalMax = max(medicalList)
  medicalMin = min(medicalList)

  minMaxList = [[foodMin, foodMax],[energyMin, energyMax], [medicalMin, medicalMax], [waterMin, waterMax]]

  def normalize(number, minimum, maximum):
    return float(number-minimum)/(maximum-minimum)

  for features in trainingFeatureVectorList:
    for i in range(len(features)-1):
      features[i] = normalize(features[i], minMaxList[i][0], minMaxList[i][1])

  #calculates similarity
  def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
      distance += pow(abs(instance1[x] - instance2[x]), 1.9)
    return pow(distance, 1/1.9)

  #gets the k nearest neighbors
  def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
      dist = euclideanDistance(testInstance, trainingSet[x], length)
      distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
      neighbors.append(distances[x][0])
    return neighbors

  #makes the decision
  def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
      response = neighbors[x][-1]
      if response in classVotes:
        classVotes[response] += 1
      else:
        classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

  def getMetrics(testSet, predictions, evaluationVector = [0,0,0,0]):
    for x in range(len(testSet)):
      if testSet[x][-1]==predictions[x]:
        if testSet[x][-1]!='None':
          evaluationVector[0]+=1
        else:
          evaluationVector[2]+=1
      else:
        if testSet[x][-1]!='None':
          evaluationVector[3] +=1
        else:
          evaluationVector[1] +=1
    tp = evaluationVector[0]
    fp = evaluationVector[1]
    tn = evaluationVector[2]
    fn = evaluationVector[3]
    total = float(sum(evaluationVector))
    accuracy = (tp+tn)/total
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return str(accuracy*100) + "%," + str(precision*100) + "%," + str(recall*100) + "%," + str(f1*100) + "%"

  #applying everything to our data
  predictions=[]
  k = 10
  evaluationVector = [0,0,0,0]
  testPrep(testingSet)
  for x in range(len(testingSet)):
    features = [0,0,0,0]
    features = getFeatures(testingSet[x], relevantWords, features)
    for i in range(len(features)-1):
      features[i] = normalize(features[i], minMaxList[i][0], minMaxList[i][1])
    neighbors = getNeighbors(trainingFeatureVectorList, features, k)
    result = getResponse(neighbors)
    predictions.append(result)
    #print('> predicted=' + repr(result) + ', actual=' + repr(testingSet[x][-1]))
  metrics = getMetrics(testingSet, predictions, evaluationVector)
  print(metrics)
