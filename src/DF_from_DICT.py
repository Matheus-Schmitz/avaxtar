import pandas as pd
import numpy as np
import pandas as pd
import pdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer
import sys



def get_files(directory,file_ext):
    file_list = []
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(file_ext):
                file_list.append(file)
    file_list.sort()
    return file_list

def get_mentions(status):
    mentionid,mentionsn = [],[]
    if len(status['entities']['user_mentions']) > 0:
        for i in range(len(status['entities']['user_mentions'])):
            mentionid.append(status['entities']['user_mentions'][i]['id'])
            mentionsn.append(status['entities']['user_mentions'][i]['screen_name'])
    return(mentionid,mentionsn)

def get_media_urls(status, tweet_type, extended_text): 
    media_urls = []
    rt_media_urls = []
    q_media_urls = []
    
    try: 
        if "extended_tweet" in status: 
            for url in status["extended_tweet"]["entities"]['media']: 
                media_urls.append(url['media_url'])
        else: 
            urls = status["entities"]["urls"]
            for url in status["entities"]['media']: 
                media_urls.append(url['media_url'])
    except: 
        pass

    try: 
        if 'retweeted_status' in status: 
            if "extended_tweet" in status["retweeted_status"]: 
                for url in status["retweeted_status"]["extended_tweet"]["entities"]['media']: 
                    rt_media_urls.append(url['media_url'])
            else: 
                for url in status["retweeted_status"]["entities"]['media']: 
                    rt_media_urls.append(url['media_url'])
    except: 
        pass

    try: 
        if 'quoted_status' in status: 
            if "extended_tweet" in status['quoted_status']: 
                for url in status["quoted_status"]["extended_tweet"]["entities"]['media']: 
                    q_media_urls.append(url['media_url'])
            else: 
                for url in status["quoted_status"]["entities"]['media']: 
                    q_media_urls.append(url['media_url'])
    except: 
        pass

    return (media_urls, rt_media_urls, q_media_urls)

def get_urls(status, tweet_type, extended_text):

    urls = []
    rt_urls = []
    q_urls = []

    if "extended_tweet" in status: 
        urls = status["extended_tweet"]["entities"]["urls"]
    else: 
        urls = status["entities"]["urls"]

    if 'retweeted_status' in status: 
        if "extended_tweet" in status["retweeted_status"]: 
            rt_urls = status["retweeted_status"]["extended_tweet"]["entities"]["urls"]
        else: 
            rt_urls = status["retweeted_status"]["entities"]["urls"]
    if 'quoted_status' in status: 
        if "extended_tweet" in status['quoted_status']: 
            q_urls = status["quoted_status"]["extended_tweet"]["entities"]['urls']
        else: 
            q_urls = status["quoted_status"]["entities"]["urls"]

    return (urls, rt_urls, q_urls)

def get_rt_urls(status,tweet_type,extended_text):
    #pdb.set_trace()
    # look through entities from the root 
    rt_urls_list = []

    if tweet_type == 'retweeted_tweet_without_comment' and extended_text == 'yes':
        try: 
#           print(count)
            rt_urls_list=status["retweeted_status"]["extended_tweet"]["entities"]["urls"]

        except:
            pass
            
    if tweet_type == 'retweeted_tweet_without_comment' and extended_text == 'no':
        try: 
#           print(count)
            rt_urls_list=status["retweeted_status"]["entities"]["urls"]

        except:
            pass
    return rt_urls_list

def get_tweet_text(status):
    rt_text = ""
    qtd_text = ""
    text = ""
    extended_text = "no"
    tweet_type = "original"
    reply_userid = None
    reply_screen = None
    reply_statusid = None

    # rt
    rt_qtd_count = 0
    rt_rt_count = 0
    rt_reply_count = 0
    rt_fav_count = 0
    rt_tweetid = None

    # quoted
    qtd_qtd_count = 0
    qtd_rt_count = 0
    qtd_reply_count = 0
    qtd_fav_count = 0
    qtd_tweetid = None

    if "extended_tweet" in status:
        text=status['extended_tweet']['full_text']
        extended_text = "yes"
    elif "text" in status: 
        text = status['text']
        extended_text = "no"

    # take care of retweets
    if 'retweeted_status' in status: 
        rt_tweetid = status['retweeted_status']['id_str']
        if 'extended_tweet' in status['retweeted_status']: 
            rt_text = status['retweeted_status']['extended_tweet']['full_text']
            extended_text = "yes"
            tweet_type = "retweeted_tweet_without_comment"
        elif 'text' in status: 
            rt_text = status['retweeted_status']['text']
            extended_text = "no"
            tweet_type = "retweeted_tweet_without_comment"

        if 'quote_count' in status['retweeted_status']:
            rt_qtd_count = status['retweeted_status']['quote_count']

        if 'retweet_count' in status['retweeted_status']:
            rt_rt_count = status['retweeted_status']['retweet_count']

        if 'reply_count' in status['retweeted_status']:
            rt_reply_count = status['retweeted_status']['reply_count']

        if 'favorite_count' in status['retweeted_status']:
            rt_fav_count = status['retweeted_status']['favorite_count']


    # take care of quoted texts
    if 'quoted_status' in status:
        qtd_tweetid = status['quoted_status']['id_str']
        if 'extended_tweet' in status['quoted_status']:
            qtd_text = status['quoted_status']['extended_tweet']['full_text']
            extended_text = "yes"
            tweet_type = "quoted_tweet"
        elif 'text' in status['quoted_status']:
            qtd_text = status['quoted_status']['text']
            extended_text = "no"
            tweet_type = "quoted_tweet"

        if 'quote_count' in status['quoted_status']:
            qtd_qtd_count = status['quoted_status']['quote_count']

        if 'retweet_count' in status['quoted_status']:
            qtd_rt_count = status['quoted_status']['retweet_count']

        if 'reply_count' in status['quoted_status']:
            qtd_reply_count = status['quoted_status']['reply_count']

        if 'favorite_count' in status['quoted_status']:
            qtd_fav_count = status['quoted_status']['favorite_count']


    if status['in_reply_to_status_id_str'] is not None and not status['truncated']:
        tweet_type = "reply"
        reply_userid = status['in_reply_to_user_id']
        reply_screen = status['in_reply_to_screen_name']
        reply_statusid = status['in_reply_to_status_id']

    elif status['in_reply_to_status_id_str'] is not None and status['truncated']:
        tweet_type = "reply"
        reply_userid = status['in_reply_to_user_id']
        reply_screen = status['in_reply_to_screen_name']
        reply_statusid = status['in_reply_to_status_id']

    return(tweet_type,text.replace("\n", " "),extended_text.replace("\n", " "), rt_text.replace("\n", " "), qtd_text.replace("\n", " "), reply_userid, reply_screen, reply_statusid, 
        rt_qtd_count, rt_rt_count, rt_reply_count, rt_fav_count, rt_tweetid, qtd_qtd_count, qtd_rt_count, 
        qtd_reply_count, qtd_fav_count, qtd_tweetid)

def get_hashtags(status,tweet_type,extended_text):
    rt_screen=''
    rt_userid=''
    q_screen=''
    q_userid=''
    rt_loc = ''
    q_loc = ''

    hashtag = []
    rt_hashtag = []
    q_hashtag = []

    if "extended_tweet" in status: 
        hashtag_list = status["extended_tweet"]["entities"]["hashtags"]
    else: 
        hashtag_list = status["entities"]["hashtags"]

    
    if len(hashtag_list)>=1:
        for i in hashtag_list:
            hashtag.append(i["text"])
    else:
        pass
            

    if tweet_type=="retweeted_tweet_without_comment":
        rt_screen=status['retweeted_status']['user']['screen_name']
        rt_userid=status['retweeted_status']['user']['id_str']
        rt_loc = status['retweeted_status']['user']['location']

        if "extended_tweet" in status["retweeted_status"]: 
            hashtag_list = status["retweeted_status"]["extended_tweet"]["entities"]["hashtags"]
        else: 
            hashtag_list = status["retweeted_status"]["entities"]["hashtags"]

        if len(hashtag_list)>=1:
            for i in hashtag_list:
                rt_hashtag.append(i["text"])
        else:
            pass
            

    elif tweet_type=="quoted_tweet":
        try: 
            q_screen=status['quoted_status']['user']['screen_name']
            q_userid=status['quoted_status']['user']['id_str']
            q_loc = status['quoted_status']['user']['location']
        except: 
            pass
        try:
            if "extended_tweet" in status["quoted_status"]: 
                hashtag_list = status["quoted_status"]["extended_tweet"]["entities"]["hashtags"]
            else: 
                hashtag_list = status["quoted_status"]["entities"]["hashtags"]

            
            if len(hashtag_list)>=1:
                for i in hashtag_list:
                    q_hashtag.append(i["text"])
            else:
                pass
                
        except:
            pass

    return(hashtag,rt_userid,rt_screen, rt_hashtag, rt_loc, q_userid,q_screen, q_hashtag, q_loc)


def get_profile_image(status):
    try:
        profile_pic_url = status['user']['profile_image_url']
        profile_banner_url = status['user']['profile_banner_url']
    except:
        profile_pic_url = ''
        profile_banner_url = ''
    return(profile_pic_url,profile_banner_url)

def main(list_of_dicts):

    fields=['tweetid','userid','screen_name','date','lang','location', "place_id", "place_url", "place_type", \
        "place_name", "place_full_name", "place_country_code", "place_country", "place_bounding_box", 'text','extended','coord', 'reply_userid', 'reply_screen', 'reply_statusid',\
        'tweet_type', "friends_count", "listed_count", "followers_count", "favourites_count", \
        "statuses_count", "verified", "hashtag", 'urls_list','profile_pic_url', 'profile_banner_url',  \
        'display_name', 'date_first_tweet', 'account_creation_date', 'rt_urls_list','mentionid',\
        'mentionsn','rt_screen','rt_userid', 'rt_text', 'rt_hashtag', 'rt_qtd_count', 'rt_rt_count', \
        'rt_reply_count', 'rt_fav_count', 'rt_tweetid', 'rt_location', 'qtd_screen','qtd_userid', 'qtd_text', 'qtd_hashtag', \
        'qtd_qtd_count', 'qtd_rt_count', 'qtd_reply_count', 'qtd_fav_count', 'qtd_tweetid', 'qtd_urls_list', 'qtd_location', 'sent_vader', \
        "token", "media_urls", "rt_media_urls", "q_media_urls"]
    
    data_frame = []

    stop_words = set(stopwords.words('english')) 
    stop_words.add("&amp;")
    stop_words.add("…")

    count = 0
    language_dist = {}
    curr_ids = []
    analyser = SentimentIntensityAnalyzer()
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)

    for status in list_of_dicts:
                
        tweetid=status['id_str']

        screen_name=status['user']['screen_name']
        userid=status['user']['id_str']
        date=status['created_at']
        lang=status['lang']

        place_id = None
        place_url = None
        place_type = None
        place_name = None
        place_full_name = None
        place_country_code = None
        place_country = None
        place_bounding_box = None
        place = status['place']
        if place is not None: 
            place_id = place['id']
            place_url = place['url']
            place_type = place['place_type']
            place_name = place['name']
            place_full_name = place['full_name']
            place_country_code = place['country_code']
            place_country = place['country']
            place_bounding_box = place['bounding_box']

        location=status['user']['location']
        coord=status['coordinates']
        friends_count=status['user']['friends_count']
        listed_count=status['user']['listed_count']
        followers_count=status['user']['followers_count']
        favourites_count=status['user']['favourites_count']
        statuses_count=status['user']['statuses_count']
        verified=status['user']['verified']
        display_name = status['user']['name']
        date_first_tweet = status['created_at']
        account_creation_date = status['user']['created_at']
        profile_pic_url,profile_banner_url=get_profile_image(status)

        # add quoted and retweet tweet ids 

        tweet_type,text,extended_text, rt_text, qtd_text, reply_userid, reply_screen, \
        reply_statusid, rt_qtd_count, rt_rt_count, rt_reply_count, rt_fav_count, rt_tweetid,\
        qtd_qtd_count, qtd_rt_count, qtd_reply_count, qtd_fav_count, qtd_tweetid  = get_tweet_text(status)

        urls_list, rt_urls_list, qtd_urls_list = get_urls(status,tweet_type,extended_text)
        #rt_urls_list=get_rt_urls(status,tweet_type,extended_text)
        mentionid,mentionsn=get_mentions(status)
        hashtag, rt_userid,rt_screen, rt_hashtag, rt_loc, qtd_userid, qtd_screen, qtd_hashtag, qtd_loc = get_hashtags(status, tweet_type, extended_text) 

        media_urls, rt_media_urls, q_media_urls = get_media_urls(status, tweet_type, extended_text)

        # vader sentiment analysis here
        comp_tweet = "" 
        if tweet_type == "quoted_tweet": 
            # do retweet text first

            try: 
                if "RT @" not in text: 
                    comp_tweet += text 
                else: 
                    pass
            except: 
                pdb.set_trace()

            try: 
                if rt_text: 
                    comp_tweet += (" " + rt_text)

                comp_tweet += (" " + qtd_text)

            except:
                pdb.set_trace()

        elif tweet_type == "retweeted_tweet_without_comment":
            comp_tweet += rt_text


        else: 
            if tweet_type != "original" and tweet_type != "reply":
                print ("something may be wrong here")
                pdb.set_trace()
            comp_tweet += text


        sent_vader = analyser.polarity_scores(comp_tweet)

        # tokenize the comp_tweet

        token_tweet = comp_tweet.replace("RT", "")
        token_tweet = token_tweet.replace("#", "")
        token_tweet = token_tweet.replace("&amp;", "")
        token_tweet = token_tweet.replace("…", "")
        token_tweet = token_tweet.replace (",", " ")
        token_tweet = token_tweet.replace("\n", " ")
        token_tweet = token_tweet.replace("!", "")
        token_tweet = token_tweet.replace(":", "")
        token_tweet = token_tweet.replace("(", " ")
        token_tweet = token_tweet.replace(")", " ")
        tokens = token_tweet.split()

        filtered_tweet = [w for w in tokens if not w in stop_words] 
        tokenized_tweet = []
        for w in filtered_tweet: 
            if "https//t.co" not in w: 
                tokenized_tweet.append(w)
        tokenized_tweet = (' '.join(tokenized_tweet) ).lower()
        tokenized_tweet = tokenized_tweet.replace("/", " ")
        #tokenized_tweet = tokenized_tweet.replace(".", " ")
        tokenized_tweet = tokenized_tweet.replace('"', " ")
        tokenized_tweet = tokenized_tweet.replace('?', " ")
        #tokenized_tweet = tknzr.tokenize(' '.join(tokenized_tweet))

        row=[tweetid,userid,screen_name,date,lang,location, place_id, place_url, place_type, place_name, place_full_name, 
        place_country_code, place_country, place_bounding_box, text,extended_text,coord, reply_userid, reply_screen, reply_statusid, 
        tweet_type,friends_count,listed_count,followers_count,favourites_count,statuses_count,verified,hashtag, urls_list, profile_pic_url,profile_banner_url, 
        display_name, date_first_tweet,account_creation_date,rt_urls_list,mentionid,mentionsn,rt_screen,rt_userid, rt_text, rt_hashtag, rt_qtd_count, rt_rt_count,
            rt_reply_count, rt_fav_count, rt_tweetid, rt_loc, qtd_screen,qtd_userid, qtd_text, qtd_hashtag, 
            qtd_qtd_count, qtd_rt_count, qtd_reply_count, qtd_fav_count, qtd_tweetid ,qtd_urls_list, qtd_loc, sent_vader['compound'], tokenized_tweet, media_urls, rt_media_urls, q_media_urls] 

        data_frame.append(row)

        count +=1

    print(f"Successfully parsed {count} tweets.")
    df = pd.DataFrame(data_frame, columns=fields)
    #print(df)
    return df