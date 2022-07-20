import pandas as pd
import numpy as np

import csv
import T_preprocessing


data = pd.read_csv("tweetstream.csv",encoding='latin1') 

data.head()

PN = data.Tweet_content

with open('tweet_preprocessed.csv', 'w', newline='') as csvfile:
    fieldnames = ['tweet_id', 'tweet_comment']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range (0,len(PN)):
        preprocessed_data = T_preprocessing.preprocess_tweet(data.Tweet_content[i])
        writer.writerow({'tweet_id': i, 'tweet_comment': preprocessed_data })

print("Preprocessing Done: 'tweet_preprocessed.csv' is generated")
