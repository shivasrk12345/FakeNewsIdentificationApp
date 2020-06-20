
from sklearn.metrics import classification_report


#df['title'].iloc[4]

################################ packages #################################################
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

#########################################################################################


#loading the dataset
df = pd.read_csv(r"C:/Users/gant0006/Desktop/Education/Data Mining/youtube final project/Ceegle-search-youtube/Dataset/news_dataset.csv", encoding = "ISO-8859-1")
df_news = df[['title','label']]
print("The shape of the loaded datatset:"+ str(df_news.shape))
#shuffle the data
df_news=df_news.sample(frac=1)
# fill the null values
df_news.title.fillna("", inplace=True)
# preprocessing the data
def ApplyPreprocessing(inputtext):
    inputtext = inputtext.translate(str.maketrans('', '', string.punctuation))
    li = []
    for word in inputtext.split():
        if word.lower():
            if word not in stopwords.words('english'):
                li.append(word)
        else:
            if(word.lower() not in stopwords.words("english")):
                li.append(word.lower())
    string1 = ""
    for i in li:
            stemmer1=SnowballStemmer("english")
            string1+=(stemmer1.stem(i))+' '
    return(string1)

#feature engineering

featureslist=df_news["title"].copy()
featureslist=featureslist.apply(ApplyPreprocessing)
vectorizer=TfidfVectorizer("english")
featureslist=vectorizer.fit_transform(featureslist)
print(type(featureslist))
featureslist=pd.DataFrame(featureslist.toarray())
print(type(featureslist))
X_train, X_test, y_train, y_test = train_test_split(featureslist, df_news['label'], test_size=0.25)
print(type(featureslist))
print(type(X_train));
# converting into list of lists
X_train  = X_train.values.tolist()
X_test= X_test.values.tolist()
y_train=list(y_train)
y_test=list(y_test);


#   implementaion of Naive Bayes classifier algorithm
class MultinomialNaiveBayesClassifier:

    def __init__(self):
        self.newscategorydict = {}  # newscategorydict for storing freq of key of news type
        self.categories = None  # for storing news categories(real/fake)

    # training the model
    def fit(self, X_train, Y_train):
        self.categories = set(Y_train)
        for i in range(len(self.categories)):
            self.newscategorydict[self.categories[i]] = {}
            for i in range(len(X_train[0])):
                self.newscategorydict[self.categories[i]][i] = 0
            self.newscategorydict[self.categories[i]]['sum'] = 0
            self.newscategorydict[self.categories[i]]['points1'] = 0
        self.newscategorydict['points1'] = len(X_train)

        for i in range(len(X_train)):
            rowlength=len(X_train[0])
            for j in range(rowlength):
                self.newscategorydict[Y_train[i]]['points1'] =self.newscategorydict[Y_train[i]]['points1']+ 1

                self.newscategorydict[Y_train[i]][j] = self.newscategorydict[Y_train[i]][j]+ X_train[i][j]


    def FindingProbability(self, testingpoint, category):
        logarithmprobabilityval=np.log(self.newscategorydict[category]['points1'])-np.log(
            self.newscategorydict['points1'])
        sum_words=len(testingpoint)
        for i in range(len(testingpoint)):
            current_word_prob=testingpoint[i] * (
                    np.log(self.newscategorydict[category][i] + 1) - np.log(
                self.newscategorydict[category]['sum'] + sum_words))
            logarithmprobabilityval+=current_word_prob
        return logarithmprobabilityval

    def predictone(self, testingpoint):
        bestclassobtained = None
        highprob= None
        startexecution = True
        for i in range(len(self.categories)):
            probOfpresentClass = self.FindingProbability(testingpoint,self.categories[i])
            if((startexecution) or (probOfpresentClass>highprob)):
                bestclassobtained=self.categories[i]
                highprob=probOfpresentClass
                startexecution = False
        return bestclassobtained

    # prediction method
    def predict(self, X_test):
        Y_pred = []
        for k in range(len(X_test)):
            Y_pred.append( self.predictone(X_test[k]) )
        return Y_pred

    # method for finding the accuracy
    def score(self, Y_pred, Y_true):
        predictedtruecount = 0
        for j in range(len(Y_pred)):
            if(Y_pred[j] == Y_true[j]):
                predictedtruecount += 1
        scoreval=predictedtruecount/len(Y_pred)
        return scoreval;



# creating object for naive bayes classifier
MNBmodelobj = MultinomialNaiveBayesClassifier()
MNBmodelobj.fit(X_train, y_train)  # training the model
Y_test_pred = MNBmodelobj.predict(X_test)  # predicting the model
our_score_test = MNBmodelobj.score(Y_test_pred, y_test)  # finding the scores
print("accuracy score is :", our_score_test)

print('classification report is:\n', classification_report(y_test, Y_test_pred))

# KNN classifier algorithm implementation from scratch

'''
import math;
from math import sqrt
def cosine_similarity(vec1, vec2):
    v11 = 0
    v12 = 0
    v22 = 0

    for i in range(len(vec1)):
        v11 += (vec1[i] * vec1[i])
        v22 += (vec2[i] * vec2[i])
        v12 += (vec1[i] * vec2[i])

    cosine_sim = (v12 / math.sqrt(v11 * v22))

    return (cosine_sim)


def euclidean_distance(vec1, vec2):
    dist = 0
    for i in range(len(vec1) - 1):
        # (x1-y1)^2
        diff = vec1[i] - vec2[i]
        dist += pow((diff), 2)
    distance = sqrt(dist)
    return distance


# finding the most similar neighbors
def get_near_neibr(train, test_row, num_neibr):
    distances = []
    neighbors = []

    # Getting all the distances
    for row in train:
        dist = euclidean_distance(test_row, row[:-1])
        distances.append((row, dist))
    # soting them to take top num_neighbor
    distances = sorted(distances, key=lambda x: x[1])

    for i in range(num_neibr):
        neighbors.append(distances[i][0])

    return neighbors


def predict_classification(train, test_row, num_neibr):
    neighbors = get_near_neibr(train, test_row, num_neibr)

    train_output = [row[-1] for row in neighbors]

    unique, counts = np.unique(train_output, return_counts=True)
    output = np.asarray((unique, counts)).T
    prediction = max(set(train_output), key=train_output.count)

    return prediction


def KNN(train_set, test_set, num_neibr):
    predictions = []
    # finding the max vote of the nearest neighbour
    for row in test_set:
        output = predict_classification(train_set, row, num_neibr)
        predictions.append(output)

    return (predictions)

'''


# implementing decision tree classifier model from scratch

'''
def trainingdataSplit(df, feature, thresholdval):
    data_vec = list()
    if ("{n}" in feature):
        featuress = sorted(df[feature].unique())

        if (len(featuress) < 2):
            return None
        for i in range(len(featuress)):
            data_vec.append(df.loc[df[feature] == featuress[i]])

    else:
        less = df.loc[df[feature] < thresholdval]
        if (len(less)):
            data_vec.append(less)
        else:
            return None

        greaterequation = df.loc[df[feature] >= thresholdval]
        if (len(greaterequation)):
            data_vec.append(greaterequation)
        else:
            return None
    return data_vec


# calculating information gain
def inform_gain(data_vec):
    total_rows = 0.0
    for i in range(len(data_vec)):
        total_rows += len(data_vec[i].index)
    proTargetVal = list()
    totalindex = np.concatenate([(sub.index) for sub in data_vec])
    UniTargetVals, targetvaluefrequency = np.unique(totalindex, return_counts=True)
    for n in range(0, len(UniTargetVals)):
        proTargetVal.append(targetvaluefrequency[n] / sum(targetvaluefrequency))

    if (proTargetVal[0] == 1):
        return 0
        # computing entropy
    entropy = 0
    for pro in proTargetVal:
        entropy += -(pro) * math.log(pro, 2)
    cond_entropy = 0
    for n, subchildframe in enumerate(data_vec):
        subchildframe_size = len(subchildframe.index)
        subchildframe_prob = subchildframe_size / total_rows
        conditional_prob = 0
        for val in [x for x in UniTargetVals if x in subchildframe.index]:
            val_freq = len([x for x in subchildframe.index == val if x == True])
            p = val_freq / subchildframe_size

            conditional_prob += -p * math.log(p, 2)

        cond_entropy += subchildframe_prob * conditional_prob
    inform_gain = entropy - cond_entropy
    if (inform_gain >= 0):
        return inform_gain
    else:
        raise Exception("negative info gain")


def EliminateNode(df):
    class1 = None
    freq = 0
    for class_val in df.index.unique():
        if (len([x for x in df.index == class_val if x == True]) > freq):
            class1 = class_val
    return class1


# Finding the best split
def bestSplit(df):
    feature, thresholdval, informationgain, data_vec = None, np.array([0]), 0, None
    for col in df.columns:
        if "{n}" in col:
            data_vec_temp = trainingdataSplit(df, col, t)
            if data_vec_temp == None:
                continue
            tempinformationgain = inform_gain(data_vec_temp)
            if (tempinformationgain > informationgain):
                informationgain = tempinformationgain
                data_vec = data_vec_temp
                feature = col
                featuress = df[col].unique()
                featuress.sort()
                thresholdval = featuress
            else:
                continue

        else:
            for t in df[col].unique():
                data_vec_temp = trainingdataSplit(df, col, t)
                if data_vec_temp == None:
                    continue

                tempinformationgain = inform_gain(data_vec_temp)

                if tempinformationgain > informationgain:
                    informationgain = tempinformationgain
                    data_vec = data_vec_temp
                    feature = col
                    thresholdval[0] = t
                else:
                    continue

    if feature == None:
        return EliminateNode(df)
    else:
        return {'feature': feature, 'value': thresholdval, 'data': data_vec}


def ApplySplitIteratively(node, max_depth, min_size, current_depth):
    childlist = list()
    for i in range(len(node['data'])):
        childlist.append(node['data'][i])
    del (node['data'])
    num_children = len(childlist)
    node['children'] = {}
    print("Depth: {}\tChildren: {}".format(current_depth, num_children))
    if current_depth >= max_depth:
        for n in range(num_children):
            node['children'][node['value'][n]] = EliminateNode(childlist[n])
        return
    if (len(node['value']) > 1):
        for child in range(num_children):
            if len(childlist[child]) > min_size:
                node['children'][node['value'][child]] = bestSplit(childlist[child])
                if (type(node['children'][node['value'][child]]) == dict):
                    ApplySplitIteratively(node['children'][node['value'][child]], max_depth, min_size,
                                          current_depth + 1)
            else:
                node['children'][node['value'][child]] = EliminateNode(childlist[child])
    else:
        if (len(childlist[0]) > min_size):
            node['children']['less'] = bestSplit(childlist[0])
            if (type(node['children']['less']) == dict):
                ApplySplitIteratively(node['children']['less'], max_depth, min_size, current_depth + 1)
            else:
                node['children']['less'] = EliminateNode(childlist[0])
        else:
            node['children']['less'] = EliminateNode(childlist[0])

        if (len(childlist[1]) > min_size):
            node['children']['greater'] = bestSplit(childlist[1])
            if (type(node['children']['greater']) == dict):
                ApplySplitIteratively(node['children']['greater'], max_depth, min_size, current_depth + 1)
            else:
                node['children']['greater'] = EliminateNode(childlist[1])
        else:
            node['children']['greater'] = EliminateNode(childlist[1])

    return


def build_decision_tree(df, max_depth, min_size):
    root = bestSplit(df)
    ApplySplitIteratively(root, max_depth, min_size, 1)
    return root


def Display(node, depth):
    if (isinstance(node, dict)):
        if ("{n}" in node['feature']):
            for n, child in enumerate(node['children']):
                print('%s[%s = %.3f]' % ((depth * ' ',
                                          (node['feature']), node['value'][n])))
                Display(node['children'][child], depth + 1)
        else:
            print('%s[%s < %.3f]' % ((depth * ' ',
                                      (node['feature']), node['value'][0])))
            for child in node['children']:
                Display(node['children'][child], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))

'''



# implementing logistic Regression classifier model from scratch
'''
zero = np.ones((X.shape[0], 1))
X = np.concatenate((zero, X), axis=1)
theta = np.zeros(X.shape[1])


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


# z is continous fucntion of x and theta

z = np.dot(X, theta)
output = sigmoid(z)


def loss(output, y):
    total_loss = (-y * np.log(output) - (1 - y) * np.log(1 - output))
    return (total_loss / X.shape[0])

    # we need to minimize the loss using gradient descent


loss_decrease = np.dot(X.T, (output - y) / X.shape[0])
learningrate = .01
theta = theta - learningrate * gradient


def predict_probability(X, theta):
    probability = sigmoid(z)
    return (probability)


def predict_values(X, theta, threshold):
    return (predict_probability(X, theta) >= threshold)


for i in range(1000):
    gradient = np.dot(X.T, (output - y)) / y.size
    theta = theta - (lr * loss_decrease)
    if (i % 10000 == 0):
        z = np.dot(X, theta)
        output = sigmoid(z)
#'''


