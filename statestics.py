from nltk import FreqDist
import pickle
import sys
##from utils import write_status
from collections import Counter
from textblob import TextBlob


def analyze_tweet(tweet):
    result = {}
    result['MENTIONS'] = tweet.count('USER_MENTION')
    result['URLS'] = tweet.count('URL')
    result['POS_EMOS'] = tweet.count('EMO_POS')
    result['NEG_EMOS'] = tweet.count('EMO_NEG')
    tweet = tweet.replace('USER_MENTION', '').replace(
        'URL', '')
    words = tweet.split()
    result['WORDS'] = len(words)
    bigrams = get_bigrams(words)
    result['BIGRAMS'] = len(bigrams)
    return result, words, bigrams


def get_bigrams(tweet_words):
    bigrams = []
    num_words = len(tweet_words)
    for i in range(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams


def get_bigram_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter



##


def get_stat(tweet):
##    tweet = "USER_MENTION recordstoredayus toms music trade URL"
##    tweet = "wind mph nne barometer in rising slowly temperature rain today in humidity"

    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    num_mentions, max_mentions = 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls = 0, 0
    num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
    num_bigrams, num_unique_bigrams = 0, 0
    all_words = []
    all_bigrams = []

    num_tweets = len(tweet.split())

    blob = TextBlob(tweet)
    for sentence in blob.sentences:
        pos_neg = sentence.sentiment.polarity
        if(pos_neg == 0):
            tweet_nat = 0
        elif(pos_neg > 0):
            num_pos_tweets = pos_neg
            tweet_nat = 1
        else:
            num_pos_tweets = pos_neg
            tweet_nat = 2


    result, words, bigrams = analyze_tweet(tweet)
    num_mentions += result['MENTIONS']
    max_mentions = max(max_mentions, result['MENTIONS'])
    num_pos_emojis += result['POS_EMOS']
    num_neg_emojis += result['NEG_EMOS']
    max_emojis = max(
        max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])

    num_urls += result['URLS']


    max_urls = max(max_urls, result['URLS'])
    num_words += result['WORDS']
    min_words = min(min_words, result['WORDS'])
    max_words = max(max_words, result['WORDS'])
    all_words.extend(words)
    num_bigrams += result['BIGRAMS']
    all_bigrams.extend(bigrams)
            
    num_emojis = num_pos_emojis + num_neg_emojis
    unique_words = list(set(all_words))
    num_unique_words = len(unique_words)
    num_unique_bigrams = len(set(all_bigrams))



    mentions =  num_mentions / float(num_tweets)
    urls =  num_urls / float(num_tweets)


    return num_pos_tweets, num_neg_tweets, mentions, urls, num_pos_emojis, num_neg_emojis, num_bigrams, tweet_nat

