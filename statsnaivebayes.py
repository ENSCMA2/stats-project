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
#imports the data and gets it ready to be processed
for x in range(0, 120):
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

  def prepData(training, relevantWords = [[],[],[],[]]):
    stop_words = set(stopwords.words('english'))
    urgency_words = ["urgent", "need", "require", "serious", "critical", "stress"]
    stemmer = PorterStemmer()
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

  def separateByClass(dataset):
  	separated = {}
  	for i in range(len(dataset)):
  		vector = dataset[i]
  		if (vector[-1] not in separated):
  			separated[vector[-1]] = []
  		separated[vector[-1]].append(vector)
  	return separated

  def getFeatures(tweet, relevantWords, featureVector = [0,0,0,0]):
      for word in tweet[0]:
        for i in range(len(relevantWords)):
          if word in relevantWords[i]:
            featureVector[i] +=1
      featureVector.append(tweet[1])
      return featureVector

  def mean(numbers):
  	return sum(numbers)/float(len(numbers))

  def stdev(numbers):
  	avg = mean(numbers)
  	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
  	return math.sqrt(variance)

  def summarize(dataset):
  	summaries = [(mean(attribute), stdev(attribute)) for attribute in dataset if type(attribute) == int]
  	#del summaries[-1]
  	return summaries

  def summarizeByClass(dataset):
  	separated = separateByClass(dataset)
  	summaries = {}
  	for classValue, instances in separated.items():
  		summaries[classValue] = summarize(instances)
  	return summaries

  def calculateProbability(x, mean, stdev):
  	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
  	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

  def calculateClassProbabilities(summaries, inputVector):
  	probabilities = {}
  	for classValue, classSummaries in summaries.items():
  		probabilities[classValue] = 1
  		for i in range(len(classSummaries)):
  			mean, stdev = classSummaries[i]
  			x = inputVector[i]
  			probabilities[classValue] *= calculateProbability(x, mean, stdev)
  	return probabilities

  def predict(summaries, inputVector):
  	probabilities = calculateClassProbabilities(summaries, inputVector)
  	bestLabel, bestProb = None, -1
  	for classValue, probability in probabilities.items():
  		if bestLabel is None or probability > bestProb:
  			bestProb = probability
  			bestLabel = classValue
  	return bestLabel

  def getPredictions(summaries, testSet):
  	predictions = []
  	for i in range(len(testSet)):
  		result = predict(summaries, testSet[i])
  		predictions.append(result)
  	return predictions

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

  trainingSet = []
  testingSet = []
  loadDataset('tweets.csv', trainingSet, testingSet)
  relevantWords = [[],[],[],[]]
  prepData(trainingSet, relevantWords)
  trainingFeatureVectorList = []
  for tweet in trainingSet:
    featureVector = [0,0,0,0]
    trainingFeatureVectorList.append(getFeatures(tweet, relevantWords, featureVector))

  summaries = summarizeByClass(trainingFeatureVectorList)
  testPrep(testingSet)
  evaluationVector = [0,0,0,0]
  testFeatures = []
  for x in range(len(testingSet)):
    features = [0,0,0,0]
    features = getFeatures(testingSet[x], relevantWords, features)
    testFeatures.append(features)

  predictions = getPredictions(summaries, testFeatures)
  metrics = getMetrics(testFeatures, predictions, evaluationVector)
  print(metrics)
