import csv
import pickle
import pandas as pd
from sklearn.svm import SVC
import T_preprocessing
import statestics

svclassifier = pickle.load(open('SVM_model.sav', 'rb'))


def get(Test_Tweet):
    classes = ['0 to 3', '4 to 7', '8 to 10']
    pred = 0

    if(Test_Tweet.find('die') != -1 or Test_Tweet.find('suicide') != -1 or Test_Tweet.find('quit') != -1):
        pred = 1

    preprocessed_Test_Tweet = T_preprocessing.preprocess_tweet(Test_Tweet)

    num_pos_tweets, num_neg_tweets, mentions, urls, num_pos_emojis, num_neg_emojis, num_bigrams, tweet_nat = statestics.get_stat(preprocessed_Test_Tweet)

    with open('Tweet_under_test.csv', 'w', newline='') as csvfile:
        fieldnames = ['num_pos_tweets', 'num_pos_tweets', 'mentions', 'urls', 'num_pos_emojis', 'num_neg_emojis', 'num_bigrams']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'num_pos_tweets':num_pos_tweets, 'num_pos_tweets':num_pos_tweets, 'mentions':mentions, 'urls':urls, 'num_pos_emojis':num_pos_emojis, 'num_neg_emojis':num_neg_emojis, 'num_bigrams':num_bigrams})

    xx = pd.read_csv("Tweet_under_test.csv")

    y_pred = svclassifier.predict(xx)


    if pred == 1:
        y_pred[0] = 2

    result = classes[y_pred[0]]

    print(result)

    return y_pred[0]
