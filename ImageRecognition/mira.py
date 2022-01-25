from __future__ import division
import util
import sys
import time
import random
from random import uniform
# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation

PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.Correct = 0
    # what values the data set can be for digits 0-9 for faces 0,1
    self.legalLabels = legalLabels
    # how many times it will loop through the training data set to 3 default
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()
    self.myWeights = []

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    #print(self.features)
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.

    """
    #call classify when ready to make prediction
    "*** YOUR CODE HERE ***"
    start = time.time() #set timer to keep track of how long it takes to train data of size inputted
    count = 0   # will be used to iterate throught trainingLabels
    myWeights = util.Counter()  # dict used to randomized weights 
    T = 0
    total = len(trainingLabels)
    """ 
    in this for loop you assign the key of myWeights to be the (x,y) 
    coordinates which corresponds to the key in each data and from there 
    you set a random weight between (-1 and 1) into myWeight[(x,y)]
    Then the last weights to be assigned is the bias weights thats given
    a random int from 1 to 3 with key being (-1,-1) to still follow the (x,y)
    pixel value and ensure it will not overwrite another key value since all numbers
    for x and y are positive integers
    """
    for data in trainingData[0]:
        myWeights[data] = round(uniform(-1,1),3)
    myWeights[(-1,-1)] = random.randint(1,3)

    """
    self.weights is a list of util.counters so self.weights[i]
    will hold the list of myWeights counters that correspond to that
    legalLabel for face ranges 0 or 1, digits 0-9
    """
    for i in self.legalLabels:
        self.weights[i] = myWeights

    """
    Here is where you take each image in trainingData and make a
    guess initially it will check if autotune is enabled if it is then
    it will run each value in Cgrid and set self.C to the value that gives
    the best percentage in order to get the most accurate prediction if
    autotune is not enabled then it will just use the default value for self.C
    after that is done the for loop will call the myGuess function and determine
    if guess was correct if it was do nothing if wrong adjust the counters
    in to corresponding label by adding /subtracting the values in data which
    is also a util.Counter
    """
    if(len(Cgrid) > 1):
        self.EnabledAutoTune(trainingData,trainingLabels,Cgrid)
    
    #x = random.sample(range(len(trainingLabels),val)
    correct = 0
    #for i in x:
    for data in trainingData:
        
        #data = trainingData[i]
        vector = self.myGuess(data)     #returns the total weighted score for each legal label
        guess = vector.argMax()         # will determine the best guess by taking largest score
        truth = trainingLabels[count]   # will set truth to the actual value for that iteration of data

        # will compare the guess key with the actual key
        if(guess == truth):

            correct += 1
        else:
            #if wrong it will do this computation to determine T or tau
            # T = the either self.C or
            # (totalweight of prediction - totalweight of truth) * avg of all features + 1 / 2*(avg of all features)^2
            # whichever has the lower value that is what T will be
            GminusT = vector[guess].__sub__(vector[truth])
            totalData = len(data)
            sizeD = data.totalCount()
            d = sizeD/totalData
            T = ((GminusT*d) + 1)/(2 * (d*d))
            T = min(self.C, T)
            TandD = util.Counter()

            # from here we get to the equation Wy = Wy - Tx and Wy' = Wy' + Tx
            # This for loop assigns the value 'Tx' for each key so that it can
            # then be subtracted or added to each weight for that key
            for key in data:
                TandD[key] = T * data[key]
            # since (-1,-1) is the bias weight it will not be a key in the data
            # so to adjust the bias you create a TandD with the same bias key and 
            # assign it a value of one so it adds 1 to the bias weight or subtracts 1
            TandD[(-1,-1)] = 1
            self.weights[guess] = self.weights[guess].__sub__((TandD))
            self.weights[truth] = self.weights[truth].__add__((TandD))

        count += 1  
    finish = time.time()   # once at this stage training is complete and time can stopped
    p = 100*correct/total
    #print("Time elapsed to train data was :" + str(round(finish-start,4)))
    #print("Guessed: " + str(correct) +"\tOut of: " + str(total) + "\t(" + str(round(p,1)) + "%)")


  """
  this helper function loops through first 100 images in training data and uses
  a different value of c from Cgrid each time then determines which values gave the best
  results sets that value to self.C to be used in actual training
  """
  def EnabledAutoTune(self,trainingData, trainingLabels, Cgrid):
      p = util.Counter()
      count = 0
      T = 0
      correct = 0
      total = len(trainingLabels)
      for c in Cgrid:
          for data in trainingData:

              vector = self.myGuess(data)
              guess = vector.argMax()
              truth = trainingLabels[count]

              if(guess == truth):

                  correct += 1
              else:

                  GminusT = vector[guess].__sub__(vector[truth])
                  totalData = len(data)
                  sizeD = data.totalCount()
                  d = total/sizeD
                  #T =((GminusT*sizeD) + 1)/(2 * (sizeD*sizeD))
                  T = ((GminusT*d) + 1)/(2 * (d*d))
                  T = min(Cgrid[0], T)
                  TandD = util.Counter()

                  for key in data:
                      TandD[key] = T * data[key]
                  TandD[(-1,-1)] = 1
                  self.weights[guess] = self.weights[guess].__sub__((TandD))
                  self.weights[truth] = self.weights[truth].__add__((TandD))
  
              if(count == 99):
                  p[c] = round(100*correct/count,2)
                  break
              count += 1
      self.C = p.argMax()

  """
  helper function used to sum list of counters in each legal label * list of counters
  in each set of data from there that gives the total weight and assigns it to vectors 
  vectors[l] is a dict with the key being legalLabels and values being total weight
  prints vectors to see how weight are currently adjusted and then returns the legalLabel with 
  highest value.
  """

  def myGuess(self, data):
      vectors = util.Counter()
      for l in self.legalLabels:
          vectors[l] = self.weights[l].__mul__(data)
          #print(self.weights[l])
          #print(vectors[l])

      return vectors
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """

    guesses = []
    for datum in data:
        vectors = util.Counter()
        for l in self.legalLabels:
            #self.weights[l] = 10
            vectors[l] = self.weights[l] * datum
            #print(vectors[l])
        guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2
    what does this MEAN!?!?!?!
    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

