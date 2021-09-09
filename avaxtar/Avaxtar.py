# Custom Modules
import avaxtar
from avaxtar import Avax_NN
from avaxtar import DF_from_DICT

# Py Data Stack
import numpy as np
import pandas as pd

# Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature Engineering
import sent2vec

# File Manipulation
from glob import glob
import joblib
import os
import gdown
#from google_drive_downloader import GoogleDriveDownloader as gdd

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# Twitter 
import tweepy
import requests

# NLP
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords



class AvaxModel():
    
    def __init__(self, consumer_key=None, consumer_secret=None, access_token=None, access_secret=None, bearer_token=None):
        super(AvaxModel, self).__init__()
        
        # Connect to Twitter
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret

        self.api_v1_connection = False
        if consumer_key and consumer_secret and access_token and access_secret:
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token,access_secret)
            self.api = tweepy.API(self.auth, retry_count=5, retry_delay=2, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
            self.api_v1_connection = True

        self.api_v2_connection = False
        if bearer_token:
            self.headers = {"Authorization": f"Bearer {bearer_token}"}
            self.api_v2_connection = True

        self.package_path = os.path.dirname(avaxtar.__file__)

        # Load sent2vec model
        if "wiki_unigrams.bin" not in os.listdir(self.package_path):
            print("Downloading sent2vec model...")
            url = 'https://drive.google.com/u/0/uc?id=0B6VhzidiLvjSa19uYWlLUEkzX3c'
            output = self.package_path + '/wiki_unigrams.bin'
            gdown.download(url, output, quiet=False)

        self.sent2vec_model = sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model('/' + self.package_path + '/' + 'wiki_unigrams.bin')#, inference_mode=True)

        # Load trained scaler
        self.scaler = joblib.load(self.package_path + '/' + 'scaler1.joblib') 

        # Tokenizer
        #self.tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
        self.stopW = stopwords.words('english')
        self.stopW.append("https://t.co")
        
    def predict_from_userid_api_v1(self, userid):
        if self.api_v1_connection:

            # Given a user ID, crawl its last 3000 tweets as a list of dictionaries
            user_timeline = [status._json for status in tweepy.Cursor(self.api.user_timeline, id=userid).items(100)]
            #print(f'User: {userid}. Timeline length: {len(user_timeline)}')

            # Extract all the features from the list of dictionaries and convert it to a datadframe
            df = DF_from_DICT.main(user_timeline) 

            # Generate timeseries features based on a user's tweets
            #timeseries_features = Feature_Engineering.prediction_timeseries_generator(df, self.scaler, sent2vec_model=self.sent2vec_model)
            timeseries_features = self.prediction_timeseries_generator_text_only(df.token.to_list(), self.scaler, sent2vec_model=self.sent2vec_model)

            # Setting the device
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Load trained model
            model = Avax_NN.AvaxNN(num_features=timeseries_features.shape[1], 
                                   learning_rate=1e-4, 
                                   optimizer=torch.optim.AdamW, 
                                   loss_fn=nn.BCELoss(), 
                                   device=device)
            model.load_state_dict(torch.load(self.package_path + '/' + 'model_pytorch1.pt', map_location=torch.device(device)))
            model.eval()

            # Send model to the device
            model = model.to(device)

            # Predict
            pred_proba = model.predict_proba(timeseries_features)
            return pred_proba

        else:
            print("In order to predict from a user id you need to input your twitter credentials to the class constructor. Please pass 'consumer_key', 'consumer_secret', 'access_token', 'access_secret'.")
    
    def predict_from_userid_api_v2(self, userid):
        if self.api_v2_connection:

            # If a screen name was passed, convert to user id
            if str(userid).isdigit() == False:
                if self.api_v1_connection: 
                    userid = self.api.get_user(userid)
                else:
                    raise ValueError("The input is not an user id. If you are trying to predict from a screen name, please connect to the v1 api.")

            # Df to store user tweets 
            df_all = pd.DataFrame()
            
            # Api v2 only allows 100 tweets per call, so we need to paginate to obtain more tweets
            pagination_token = None
            while df_all.shape[0] < 100:
                response = self.make_request(self.headers, userid, pagination_token)       
                if 'data' not in response:
                    print(response)
                df = pd.DataFrame(response['data'])
                df_all = pd.concat((df_all, df))
                try:
                    pagination_token = response['meta']['next_token']
                except:
                    break

            list_of_tweets = df_all['text'].to_list()
            #print(f'User: {userid}. Timeline length: {len(list_of_tweets)}')

            pred_proba = self.predict_from_tweets(list_of_tweets)
            return pred_proba

        else:
            print("In order to predict from a user id you need to input your twitter credentials to the class constructor. Please pass 'bearer_token'.")

    def predict_from_tweets(self, list_of_tweets):
        # Ensure tweets are tokenized
        #list_of_tweets = [' '.join(self.tknzr.tokenize(str(tweet))) for tweet in list_of_tweets]
        list_of_tweets = [' '.join([word for word in tweet.split() if word not in self.stopW and "https://t.co" not in word]) for tweet in list_of_tweets]
        list_of_tweets = [self.clean_tweet(tweet) for tweet in list_of_tweets]

        # Generate timeseries features based on a user's tweets
        timeseries_features = self.prediction_timeseries_generator_text_only(list_of_tweets, self.scaler, sent2vec_model=self.sent2vec_model)

        # Setting the device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load trained model
        model = Avax_NN.AvaxNN(num_features=timeseries_features.shape[1], 
                               learning_rate=1e-4, 
                               optimizer=torch.optim.AdamW, 
                               loss_fn=nn.BCELoss(), 
                               device=device)
        model.load_state_dict(torch.load(self.package_path + '/' + 'model_pytorch1.pt', map_location=torch.device(device)))
        model.eval()

        # Send model to the device
        model = model.to(device)

        # Predict
        pred_proba = model.predict_proba(timeseries_features)
        return pred_proba
        
    def prediction_timeseries_generator_text_only(self, list_of_tweets, scaler, sent2vec_model=None):
        # Concatenate all of a users tweets in that period, then embed them with sent2vec twitter_bigrams model
        user_tweets = ' '.join(str(x) for x in list_of_tweets)
        embedding = sent2vec_model.embed_sentence(user_tweets)

        # Merge text features (embedding), user features, url features and label
        user_data = np.concatenate([embedding[0]])
        timeseries_dataset = np.array(user_data)

        # Rescale features
        timeseries_features = scaler.transform([timeseries_dataset])
        return timeseries_features

    def clean_tweet(self, tokenized_text):
        tokenized_text = str(tokenized_text)
        tokenized_text = tokenized_text.replace("RT", "")
        tokenized_text = tokenized_text.replace("#", "")
        tokenized_text = tokenized_text.replace("&amp;", "")
        tokenized_text = tokenized_text.replace("…", "")
        tokenized_text = tokenized_text.replace (",", " ")
        tokenized_text = tokenized_text.replace("\n", " ")
        tokenized_text = tokenized_text.replace("!", "")
        tokenized_text = tokenized_text.replace(":", "")
        tokenized_text = tokenized_text.replace("(", " ")
        tokenized_text = tokenized_text.replace(")", " ")
        tokenized_text = tokenized_text.replace("/", " ")
        tokenized_text = tokenized_text.replace('"', " ")
        tokenized_text = tokenized_text.replace('?', " ")
        tokenized_text = tokenized_text.lower()
        return tokenized_text

    def make_request(self, headers, user_id, pagination_token=None):
        url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        if pagination_token:
            params = f"max_results=100&tweet.fields=id,text&pagination_token={pagination_token}"
        else:
            params = f"max_results=100&tweet.fields=id,text"
        return requests.request("GET", url, params=params, headers=headers).json()