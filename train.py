import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def wrangle(content):
    content =  content.lower()
    content =  re.sub(r'[^\x00-\x7F]+',' ', content)  
    content = re.sub(r'@[\w]+' , '' , content)
    content =  re.sub(r"https://.*?"," ",content)
    content =  re.sub(r"http://.*?"," ",content)
    content =  re.sub(r"\b(([a-z\d]+).com)\b"," ",content)
    content = re.sub(r'(\.+)|(!+)|(\?+)|(#+)|(\&+)|(\++)|(/+)|(=+)|([|]+)|(\$+)|(%+)|(\^+)|(\_+)','',content)
    content = re.sub(r' +' , ' ' ,content)
    content = re.sub(r'([a-z])\1+' , r'\1\1',content)
    content =  str.strip(content)  
    return content


def train(args):
    #Loading the data set
    TrainingData = pd.read_csv(os.path.join(args.data_path, 'training.csv'),header=None,encoding='latin-1')
    TrainingData[0] = TrainingData[0].replace([4],1).astype(float)

    #Data Preprocessing
    TrainingData[1] = TrainingData[1].apply(lambda x: wrangle(str(x)))

    #Separating samples and labels
    y_train = TrainingData[0]
    X = TrainingData[1]

    #Vectorizing
    vectorizer = TfidfVectorizer(max_df=0.9, min_df = 5, ngram_range = (1,2))
    x_train = vectorizer.fit_transform(X)
    vectorizer_file = 'vectorizer.sav'
    pickle.dump(vectorizer,open(vectorizer_file,'wb'))

    #Fitting Model 1
    print("Running Logistic Regression")
    LR = LogisticRegression(solver = 'liblinear' , penalty = 'l2')
    LR.fit(x_train, y_train) 

    #Fitting Model 2
    print("Running Linear SVM 1")
    svc = LinearSVC(max_iter = 700, C= 0.3)
    svc.fit(x_train, y_train) 

    #Fitting Model 3
    print("Running SVM 2")
    svc2 = LinearSVC(max_iter = 1000, C= 0.025)
    svc2.fit(x_train, y_train)

    #Fitting Model 4
    print("Running Naive Bayes")
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    #Fitting Model 5
    print("Running Logistic Regression 2")
    LR2 = LogisticRegression(solver = 'liblinear', penalty = 'l1')
    LR2.fit(x_train, y_train) 

    #Save all the models
    LR_saved_model = 'LRModel.sav' 
    lr_file = os.path.join(args.model_path, LR_saved_model)
    pickle.dump(LR,open(lr_file,'wb'))

    SVM1_saved_model = 'SVC1Model.sav'
    svc1_file = os.path.join(args.model_path, SVM1_saved_model)
    pickle.dump(svc,open(svc1_file,'wb'))

    SVM2_saved_model = 'SVC2Model.sav'
    svc2_file = os.path.join(args.model_path, SVM2_saved_model)
    pickle.dump(svc2,open(svc2_file,'wb'))

    NB_saved_model = 'NBModel.sav'
    nb_file = os.path.join(args.model_path, NB_saved_model)
    pickle.dump(nb,open(nb_file,'wb'))

    LR2_saved_model = 'LR2Model.sav' 
    lr2_file = os.path.join(args.model_path, LR2_saved_model)
    pickle.dump(LR2,open(lr2_file,'wb'))

    #Done

def starter():
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 Training File')
    parser.add_argument('--data_path', default='training.csv', type=str, help='Path to data file')
    parser.add_argument('--model_path' , default = 'model' , type=str, help = 'Path to saved models' )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = starter()
    train(args)
