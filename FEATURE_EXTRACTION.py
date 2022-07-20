import statestics
import pandas as pd
import numpy as np
import csv

data = pd.read_csv("tweet_preprocessed.csv",encoding='latin1') 

data.head()

PN = data.tweet_comment

with open('Tweet_Features.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_pos_tweets', 'num_pos_tweets', 'mentions', 'urls', 'num_pos_emojis', 'num_neg_emojis', 'num_bigrams', 'tweet_nat']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range (0,len(PN)):
        num_pos_tweets, num_neg_tweets, mentions, urls, num_pos_emojis, num_neg_emojis, num_bigrams, tweet_nat = statestics.get_stat(data.tweet_comment[i])
        writer.writerow({'num_pos_tweets':num_pos_tweets, 'num_pos_tweets':num_pos_tweets, 'mentions':mentions, 'urls':urls, 'num_pos_emojis':num_pos_emojis, 'num_neg_emojis':num_neg_emojis, 'num_bigrams':num_bigrams, 'tweet_nat':tweet_nat })

print("Feature Extraction Done: 'Tweet_Features.csv' is generated")


