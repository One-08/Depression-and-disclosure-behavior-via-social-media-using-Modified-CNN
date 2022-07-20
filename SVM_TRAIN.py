import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


bankdata = pd.read_csv("Tweet_Features.csv")

X = bankdata.drop('tweet_nat', axis=1)
y = bankdata['tweet_nat']

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X, y)

import pickle

pickle.dump(svclassifier, open('SVM_model.sav', 'wb'))
