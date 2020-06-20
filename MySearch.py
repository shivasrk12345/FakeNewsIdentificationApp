
# libraries for flask

from flask import Flask, redirect, url_for, request, render_template

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
import pandas as pd
import pprint
import matplotlib.pyplot as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter
#from main1 import *


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
df = pd.read_csv("Dataset/news_dataset.csv", encoding = "ISO-8859-1")
df_news = df[['title','label']]
print("The shape of the loaded datatset:"+ str(df_news.shape))
#shuffle the data
df_news = df_news.sample(frac=1)
# fill the null values
df_news.title.fillna("", inplace=True)
def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    ls = []
    for word in text.split():
        if word.lower():
            if word not in stopwords.words('english'):
                ls.append(word)
        else:
            word = word.lower()
            if word not in stopwords.words("english"):
                ls.append(word)
    words = ""
    for i in ls:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+ " "
    return(words)
def vectorizer_1():
    features   = df_news["title"].copy()
    features   = features.apply(pre_process)
    vectorizer = TfidfVectorizer("english")
    features   = vectorizer.fit_transform(features)
    return (vectorizer)



def youtube_search(q, max_results=25,order="relevance", token=None, location=None, location_radius=None):
    DEVELOPER_KEY = "AIzaSyAcdAZVrDJAP6YOQFHsOQy8069LEYEZHpI"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(
    q=q,
    type="video",
    pageToken=token,
    order = order,
    part="id,snippet", # Part signifies the different types of data you want
    maxResults=max_results,
    location=location,
    locationRadius=location_radius).execute()

    title = []
    channelId = []
    channelTitle = []
    categoryId = []
    videoId = []
    viewCount = []
    likeCount = []
    dislikeCount = []
    commentCount = []
    favoriteCount = []
    category = []
    tags = []
    videos = []

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            title.append(search_result['snippet']['title'])
            videoId.append(search_result['id']['videoId'])
            response = youtube.videos().list(part='statistics, snippet',id=search_result['id']['videoId']).execute()
            channelId.append(response['items'][0]['snippet']['channelId'])
            channelTitle.append(response['items'][0]['snippet']['channelTitle'])
            categoryId.append(response['items'][0]['snippet']['categoryId'])
            favoriteCount.append(response['items'][0]['statistics']['favoriteCount'])
            viewCount.append(response['items'][0]['statistics']['viewCount'])
            #likeCount.append(response['items'][0]['statistics']['likeCount'])
            #dislikeCount.append(response['items'][0]['statistics']['dislikeCount'])

        if 'commentCount' in response['items'][0]['statistics'].keys():
            commentCount.append(response['items'][0]['statistics']['commentCount'])
        else:
            commentCount.append([])

        if 'tags' in response['items'][0]['snippet'].keys():
            tags.append(response['items'][0]['snippet']['tags'])
        else:
            tags.append([])

    youtube_dict = {'tags':tags,'channelId': channelId,'channelTitle': channelTitle,'categoryId':categoryId,'title':title,'videoId':videoId,'viewCount':viewCount,'likeCount':likeCount,'dislikeCount':dislikeCount,'commentCount':commentCount,'favoriteCount':favoriteCount}

    return(youtube_dict["title"])

def results_displayed(output):
    #output = youtube_search("black friday deals")
    ls = {}
    title = ['title']
    #print(output)
    #df = pd.DataFrame(columns = title)

    #df.to_csv(r'/Users/chanukya/Documents/GitHub/DataMining/output1.csv')
    #output.to_csv (r'/Users/chanukya/Documents/GitHub/DataMining/output.csv', index = None, header=True)
    loaded_model = pickle.load(open('model.py', 'rb'))
    vectorizer = vectorizer_1()
    df = pd.DataFrame(columns=title)
    df['title'] = output
    values   = df["title"].copy()
    values   = values.apply(pre_process)
    #vectorizer = TfidfVectorizer("english")
    values   = vectorizer.transform(values)
    result = loaded_model.predict(values);
    for i in range(df.shape[0]):
        ls[df['title'].iloc[i]] = result[i]
    return(ls)







def final_prediction(searchword):
    output = youtube_search(str(searchword))
    return results_displayed(output)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('search_page.html')

@app.route('/search/<search_key>/')
def success(search_key):
    youtube_search_df = final_prediction(search_key)
    print(youtube_search_df);

    '''pred = list(google_search_df['pred'])
    ref = list(google_search_df['href'])
    title = list(google_search_df['title'])'''

    return render_template('result_page.html', search_key=search_key, output=youtube_search_df,length=len(youtube_search_df))




@app.route('/search', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        search_key = request.form['search_key']
        print(search_key)
        return redirect(url_for('success', search_key = search_key))
    else:
        search_key = request.form['search_key']
        return redirect(url_for('success', search_key = search_key))

if __name__ == '__main__':
    app.run(debug=True)
