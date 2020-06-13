# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:05:51 2019

@author: raghe
"""
from py4j.java_gateway import JavaGateway
import tweepy
import traceback

from toolkit import *
import pandas as pd
import json
from keras.models import load_model
from keras.preprocessing import text, sequence
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import pickle
import numpy as np
from keras import backend as K



class Tweet:
    """A single tweet that need classification."""

    def __init__(self, tid, text, label=None, JSON=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the tweet.
            text_a: string. The untokenized text of the tweet
            url: tweet url
            labels: (Optional) [int]. The label of the tweet, should be specifed by the predictor.
        """
        self.tid = tid
        self.text = text
        self.lemma = None
        self.preprocessed = None
        self.seq = None
        self.label = label
        self.prediction = None
        self.JSON = JSON
                
    def basic_preprocess(self):
        
        if self.preprocessed != None:
            print("Already processed .. ")
            
        with open(r"stop_words.csv" , 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()     
            stopwords = [line.strip() for line in lines]   
        with open(r"hashtags.txt" , 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            hashtags = [line.strip() for line in lines]

        self.preprocessed = remove_hashtags(self.text ,hashtags)
        self.preprocessed = remove_cosuctive_letters(self.preprocessed)
        self.preprocessed = replace_emoji_with_description(self.preprocessed)
        self.preprocessed = normlize_twitter_spesfice_tokens(self.preprocessed)
        self.preprocessed = deNoise(self.preprocessed)
        self.preprocessed = cleaning(self.preprocessed)
        self.preprocessed = normlization(self.preprocessed)
        self.preprocessed = remove_stopwords(self.preprocessed, stopwords)
        self.preprocessed = remove_non_arabic_letteres(self.preprocessed)

    
    def set_prediction(self , prediction):
        self.prediction = prediction
        self.label = "Hate" if (int(round(prediction[0])) == 1) else "Normal" 
        

class MyStreamListener(tweepy.StreamListener):
    
    
    def __init__(self ,  dataframe_path, batch_size = 10000 , starting_file_no = 0):
        super(MyStreamListener,self).__init__()
        self.i = starting_file_no * batch_size
        self.dataframe_path = dataframe_path
        self.batch_size = batch_size
        self.list_of_dic = []

    def on_status(self, tweet):

        self.i = self.i + 1
         #will hold tweets as dic

        tweet = tweet._json
        if not 'retweeted_status' in tweet:
            if 'extended_tweet' in tweet:
                tweet_text = tweet['extended_tweet']['full_text']
            else:
                tweet_text = tweet['text']
        else:
            return 
        
        dic = { 
        "created_at" : tweet["created_at"],
        "id_str": tweet["id_str"],
        "text" : tweet_text,
        "screen_name": tweet["user"]["screen_name"],
        "user_id": tweet["user"]["id_str"],
        "location": tweet['user']["location"],
        "hashtags" : 'None' if len(tweet['entities']["hashtags"]) == 0 else " , ".join([hashtag['text'] for hashtag in tweet['entities']["hashtags"]])
        }
        self.list_of_dic .append(dic)

        if self.i % self.batch_size == 0:
            
            #print("Writing file {}.. ".format(int(self.i/self.batch_size)))
            df = pd.DataFrame.from_dict(self.list_of_dic )
            current_path = self.dataframe_path + "-{}.csv".format(int(self.i/self.batch_size))
            df.to_csv(current_path)
            self.list_of_dic  = [] #Reset               
                               




class DataStreamer:
    
    def __init__(self):
        self.consumer_key = ''
        self.consumer_secret = ''
        self.access_token = ''
        self.access_token_secret = ''
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.data = [] #list of tweets

    
    def start_streaming_tweets(self, keyword, dataframe_path, batch_size = 10000, starting_file_no = 0):
        #
        myStreamListener = MyStreamListener(dataframe_path, batch_size, starting_file_no)
        myStream = tweepy.Stream(auth = self.api.auth, listener=myStreamListener)
        while True:
            try:
                keys = keyword.split()
                myStream.filter(track=keys)

            except Exception as e:
                print(e)
                print(traceback.format_exc())
                continue


    def start_sampling_tweets(self):
        myStreamListener = MyStreamListener()
        myStream = tweepy.Stream(auth = self.api.auth, listener=myStreamListener)
        while True:
            try:
		
                myStream.sample()
            except Exception as e:
                print(e)
                print("error")
                continue 
    


class DataRetriver:
    
    def __init__(self, MAX=None):
        """Constructs a DataRetriver instance.
        
        Args:

        """
    
        self.consumer_key = 'iv9fgInq9KtsGYkGpHYDsIl0E'
        self.consumer_secret = 'vjLoyMjRi8AVdtqstkitQcdQoEBLDjrFdkCBNdYE3G31ZX603Y'
        self.access_token = '113729650-uIF1Mr1RnQFS5SVvrZSygdb3yoCiDgFMd6Ipe1aL'
        self.access_token_secret = 'QhBGXLWhbIAQSDmfaEkt3s6sF9aDLGhksOS0XuBWVCFFe'
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.data = [] #list of tweets
        self.MAX = MAX


    def get_tweets_from_keyword_cont(self,keywords, batch, MAX=0): #always json
       
        while(True):
            try:
            
                i = 0 #count tweets
                cont = MAX #count files
                List = keywords.split() #prepare keywords 
                keywords = " OR ".join(List)
                
                for tweet in tweepy.Cursor(self.api.search, q= keywords + "  -filter:retweets", tweet_mode='extended',count=100,
                                                   lang="ar",
                                                   since="2019-01-01").items():
                    
    
                    i = i + 1 #incremet tweets number once new tweet is retrived 
                    tmp =  Tweet(tweet.id , tweet.full_text, JSON= tweet._json ) #create tweet object
                    self.data.append(tmp)
                    

                    if (i%batch == 0): # save every batch in new file, numbered by con
                        cont = cont + 1
                        list_of_dic = []
                        for tweet in self.data:
                            tweet = tweet.JSON
                            if 'full_text' in tweet:
                                text = tweet['full_text']
                            else:
                                text = tweet['text']
                            
                            dic = { 
                            "created_at" : tweet["created_at"],
                            "id_str": tweet["id_str"],
                            "text" : text,
                            "screen_name": tweet["user"]["screen_name"],
                            "user_id": tweet["user"]["id_str"],
                            "location": tweet['user']["location"],
                            "hashtags" : 'None' if len(tweet['entities']["hashtags"]) == 0 else " , ".join([hashtag['text'] for hashtag in tweet['entities']["hashtags"]])
                            }
                            list_of_dic.append(dic)
                    
                        df = pd.DataFrame.from_dict(list_of_dic)
                        file_path = 'D:\IR\ProCollection\Collection\corona\coronadata-{}.csv'.format(cont)
                        df.to_csv(file_path)
                        self.data = []
                        
            
            except Exception as e:
                continue 

            
    def get_tweets_from_keyword(self,keywords, JSON = False):
       
        while(True):
            try:
                i = 0 
                List = keywords.split() 
                keywords = " OR ".join(List)
                
                for tweet in tweepy.Cursor(self.api.search, q= keywords + "  -filter:retweets", tweet_mode='extended',count=100,
                                                   lang="ar",
                                                   since="2019-01-01").items():
    
                    i = i + 1
                    #if (i%100 == 0):
                        #print("{} - Retried 100 tweets ..".format(i))
                    if not (JSON):
                        tmp =  Tweet(tweet.id , tweet.full_text)
                        self.data.append(tmp)
                    else:
                        tmp =  Tweet(tweet.id , tweet.full_text, JSON= tweet._json )
                        self.data.append(tmp)
                    

                    if (i == self.MAX):
                        return
    
    
    
            except Exception as e:
                continue
            
    
    def get_tweets_from_username(self,screen_name, JSON = False) :
        
        
        try:
            i = 0 

            for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=screen_name, tweet_mode="extended", count=100).items():
                
                i = i + 1
                
                if not (JSON):
                    tmp =  Tweet(tweet.id , tweet.full_text)
                    self.data.append(tmp)
                else:
                    tmp =  Tweet(tweet.id , tweet.full_text, JSON= tweet._json )
                    self.data.append(tmp)
                if (i == self.MAX):
                    break



        except Exception as e:
            print(e)
            print("error")
            
    def get_trends(self) :
        
        
        try:

            trends1 = self.api.trends_place(23424938) # from the end of your code
            # trends1 is a list with only one element in it, which is a 
            # dict which we'll put in data.
            data = trends1[0] 
            # grab the trends
            trends = data['trends']
            # grab the name from each trend
            names = [trend['name'] for trend in trends]
            # put all the names together with a ' ' separating them
            
            return names

        except Exception as e:
            print(e)
            print("error")  
            return None
			
			

              
class DataProcessor:
    
    def __init__(self, data):
        """Constructs a DataRetriver instance.   
        Args:
        """
        self.data = data #list of tweets - pass by value
        self.is_preprocessed = False
        
    def preprocess(self):

        if(type(self.data[0]) == 'Tweet'):
            print('Not supported, Please make sure that your data is a list of Tweet objects')
            return
           
        print("Preprocessing ... ")
        print(len(self.data))
        
        for tweet in self.data:
            tweet.basic_preprocess()
            
        self.is_preprocessed = True
        
        print("lemmatize all ... ")
        self._lemmatize()
        print("Done lemmatize all ... ")
    
    def _lemmatize(self):
        
        frasa_gateway = FrasaGateway()
        #make sure it has preprcessed
        lemmas = frasa_gateway.LemmatizeAll([tweet.preprocessed for tweet in self.data]) 
        
        #set lemmas to tweets:
        for i, tweet in enumerate(self.data):
            tweet.lemma = lemmas[i]
    
  
        
            
class FrasaPOSGateway: 
    def __init__(self):
        self.gateway = JavaGateway(auto_convert=True)
        self.POSTagger = self.gateway.entry_point.getPOSTagger()
        
    def POSTagAll(self,text_list):
        tweet = self.POSTagger.POSTagAll(text_list)
        return tweet        
                    

  
                  
class FrasaGateway: 
    def __init__(self):
        self.gateway = JavaGateway(auto_convert=True)
        self.Lemmatizer = self.gateway.entry_point.getLemmatizer()
        
    def LemmatizeAll(self,text_list):
        tweet = self.Lemmatizer.LemmatizeAll(text_list)
        return tweet 
        

class predictor:

    @staticmethod
    def predict(tweets_batch):
        K.clear_session()

        model = load_model('best-model/CNN_model.h5')
        predictor.prepare_input(tweets_batch)
        inputs = np.asarray([tweet.seq for tweet in tweets_batch])
        predictions = model.predict(inputs)
        for i, tweet in enumerate(tweets_batch):
            tweet.set_prediction(predictions[i])
        
        return predictions
        
        
    @staticmethod
    def prepare_input(tweets_batch):
        with open('best-model/tokenizer.pickle', 'rb') as handle:
             token = pickle.load(handle)
        MAXLENGTH = 115
        
        
        tweet_seqs = sequence.pad_sequences(token.texts_to_sequences([tweet.lemma for tweet in tweets_batch]) , maxlen=MAXLENGTH)
        for i, tweet in enumerate(tweets_batch):
            tweet.seq = tweet_seqs[i] 
            
            
 

class detect:
    
    def start(keyword, JSON = False):
        #retrive data      
        dr = DataRetriver()
        
        if (keyword.startswith("@")):
            dr.get_tweets_from_username(keyword, JSON)
        else:
            dr.get_tweets_from_keyword(keyword, JSON)
        
        if(len(dr.data) == 0 ):
            return 0,0,0,0
        print("Retrived {} tweets .. ".format(len(dr.data)))
        #send retrived values to the processor
        dp = DataProcessor(dr.data)
        #start processing
        dp.preprocess()
        print("Processed {} tweets .. ".format(len(dr.data)))
        #predict
        pred = predictor()
        pred.predict(dp.data)
        
        #convert to list of dic
        
        hatePred = []
        nonTatePred = []
        hateTotal = 0
        nonHateTotal = 0
        
        
        for tweet in dp.data:
            tmp_dic = {}
            tmp_dic['id'] = tweet.tid
            tmp_dic['text'] = tweet.text
            tmp_dic['label'] = tweet.label
            if (tweet.label == "Hate"):
                hatePred.append(tmp_dic)
                hateTotal= hateTotal + 1
            else:
                nonTatePred.append(tmp_dic)
                nonHateTotal= nonHateTotal + 1                
                
        return hatePred, nonTatePred, hateTotal,nonHateTotal 

    '''
		data: list of Tweet objects
	'''
    def start_with_existing_data(data):
        
        #send retrived values to the processor
        dp = DataProcessor(data)
        #start processing
        dp.preprocess()
        print("Processed {} tweets .. ".format(len(data)))
        #predict
        pred = predictor()
        pred.predict(dp.data)
        
        #convert to list of dic
        
        hatePred = []
        nonTatePred = []
        hateTotal = 0
        nonHateTotal = 0
        
        
        for tweet in dp.data:
            tmp_dic = {}
            tmp_dic['id'] = tweet.tid
            tmp_dic['text'] = tweet.text
            tmp_dic['label'] = tweet.label
            tmp_dic['prediction'] = tweet.prediction[0]
            if (tweet.label == "Hate"):
                hatePred.append(tmp_dic)
                hateTotal= hateTotal + 1
            else:
                nonTatePred.append(tmp_dic)
                nonHateTotal= nonHateTotal + 1                
                
        return hatePred, nonTatePred, hateTotal,nonHateTotal 



        

class trends:
    
    def get_trends():
        dr = DataRetriver()
        trends = dr.get_trends()
        
        
        return trends 
