# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import time
import util
from random import uniform
import random

PRINT = True


class PerceptronClassifier:
    """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        self.bias = {}

        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use
            self.bias[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights == weights;

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    The training loop for the perceptron passes through the training data several

    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

        # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        #trainingData = random.shuffle(trainingData)
        self.features = trainingData[0]
        myWeights = util.Counter()
        "sets the bias weight randomly"
        myBias = round(uniform(-1, 1), 3)

        "sets bias counter label with a weight"
        for j in self.legalLabels:
            self.bias[j] = myBias

        #print "the bias" ,self.bias
        "randomizes data into a list of weights"
        for data in trainingData[0]:
            myWeights[data] = round(uniform(-1, 1), 3)

        "sets regular weights"
        for j in self.legalLabels:
            self.weights[j] = myWeights

        #x = random.sample(range(len(trainingData)), 451)
        for iteration in range(self.max_iterations):
            #print "Starting iteration ", iteration, "..."
            "each iteration true label iteration is reset"
            count = 0
            for i in range(len(trainingData)):

                "*** YOUR CODE HERE ***"
                "sets predicted label and the true label"
                predictedLabel = self.getScore(trainingData[i])
                trueLabel = trainingLabels[count]
                #print("Guessed : " + str(guess)+ "\tTruth : " + str(truth))
                "if checks the predicted label against the true label and if " \
                "the label predicted was right you would just continue"

                "else means the predicted label did not predict the true label"

                "The labels can be wrong based in two ways. Either you predicted a label 0(notface/notdigit)" \
                " but it was really label 1(face/digit). " \
                "Or you predicted label 1(face/digit) but you got label 0 (notface/notdigit)." \
                " The regular weights and the bias weights are updated based on the true labels " \
                "and the predicted labels."
                if trueLabel == predictedLabel:
                    "right"

                else:
                    "wrong"
                "updates weights from true label by"
                "adding the features to the weights i.e rewards the label's score"
                "we want this label to happen"
                self.weights[trueLabel] = self.weights[trueLabel].__add__(trainingData[i])
                "updates bias weight for true label by adding 1 i.e reward the label's score"
                self.bias[trueLabel] += 1

                #print self.weights[truth]
                "updates weights for predicted label weights by subtracting from features"
                "we want this label not to happen i.e punish the label's score"
                self.weights[predictedLabel] = self.weights[predictedLabel].__sub__(trainingData[i])
                "updates bias weight for predicted label by subtracting 1 i.e punishes the label's score"
                self.bias[predictedLabel] -= 1
                #print self.weights[predictedScore]
                "incrementing of count of through true labels"
                count += 1
                #print count
            #print "training data",i

            "this gets the score of each data point by getting the dot product of the weights and features of each legal label." \
            "and adding the bias weight to each legal label"
            "It then returns the max label of the two labels based on the score"
    def getScore(self, i):

            score = util.Counter()

            for l in self.legalLabels:

                score[l] = self.weights[l].__mul__(i)
                #print self.bias
                #print "the first score ", score
                score[l] = score[l] + self.bias[l]
                #print "when bias added ", score

                predictedLabel = score.argMax()

                #print "final score ",finalScore

            return predictedLabel



    def classify(self, data):
        """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """

        guesses = []
        for datum in data:

            vectors = util.Counter()

            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum


            guesses.append(vectors.argMax())
        return guesses
