from __future__ import division
from lib import *
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
from collections import Counter
#imports the data and gets it ready to be processed

for x in range(0, 160):

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
    stemmer = PorterStemmer()
    for tweet in testing:
      for word in word_tokenize(tweet[0]):
        if word[0]=="@":
          tweet[0] = tweet[0].replace(word, "")
      for c in "!@#%&*()[]{-}/?<\">":
        tweet[0] = tweet[0].replace(c, "").lower()
      tweet[0] = [w for w in word_tokenize(tweet[0]) if not w in stop_words]
      for word in tweet[0]:
        word = stemmer.stem(word)

  def featurize(tweet):
      features = Counter()
      for i in tweet:
      	features[i] += 1
      return features

  trainingSet = []
  testingSet = []
  loadDataset('tweets.csv', trainingSet, testingSet)
  relevantWords = [[],[],[],[]]
  prepData(trainingSet, relevantWords)
  testPrep(testingSet)
  train_data = []
  for tweet in trainingSet:
      train_data.append([featurize(tweet[0]), tweet[1]])

  classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
  sorted(classifier.labels())
  test_data = [featurize(test_tweet[0]) for test_tweet in testingSet]
  predictions = classifier.classify_many(test_data)

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

  evaluationVector = [0,0,0,0]

  metrics = getMetrics(testingSet, predictions, evaluationVector)
  print(metrics)
