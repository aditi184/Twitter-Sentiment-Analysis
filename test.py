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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

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


def test(args):
    #loading input and output file
    testing_data = pd.read_csv(args.input_file , header = None, encoding = 'latin-1')
    
    #preprocessing the inputfile
    testing_data = testing_data[0].apply(lambda x: wrangle(str(x)))

    #Vectorizing the input
    vectorizer_file = 'vectorizer.sav'
    vectorizer = pickle.load(open(vectorizer_file,'rb'))
    x_test = vectorizer.transform(testing_data)

    #loading the saved models
    model_path = args.model_directory

    LR_saved_model = 'LRModel.sav' 
    LR_model_file = os.path.join(model_path, LR_saved_model)
    LR = pickle.load(open(LR_model_file, 'rb'))

    SVM1_saved_model = 'SVC1Model.sav'
    SVM1_model_file = os.path.join(model_path,SVM1_saved_model)
    SVM1 = pickle.load(open(SVM1_model_file, 'rb'))

    SVM2_saved_model = 'SVC2Model.sav'
    SVM2_model_file = os.path.join(model_path,SVM2_saved_model)
    SVM2 = pickle.load(open(SVM2_model_file, 'rb'))

    NB_saved_model = 'NBModel.sav'
    nb_file = os.path.join(model_path, NB_saved_model)
    NB = pickle.load(open(nb_file,'rb'))

    LR2_saved_model = 'LR2Model.sav' 
    LR2_model_file = os.path.join(model_path, LR2_saved_model)
    LR2 = pickle.load(open(LR2_model_file, 'rb'))


    #making predictions
    P1 = LR.predict(x_test)
    S1 = pd.Series(P1)

    P2 = SVM1.predict(x_test)
    S2 = pd.Series(P2)

    P3 = SVM2.predict(x_test)
    S3 = pd.Series(P3)

    P4 = NB.predict(x_test)
    S4 = pd.Series(P4)

    P5 = LR2.predict(x_test)
    S5 = pd.Series(P5)

    allPreds = { 'LR': S1, 'SVC1': S2 , 'SVC2' : S3 , 'NB' : S4 , 'LR2' : S5 }
    df = pd.DataFrame(allPreds)
    FinalPreds = df.mode(axis=1)

    #writing output file 
    FinalPreds = FinalPreds.replace([1],4).astype(int)
    FinalPreds.to_csv(args.output_file, index=False, header=False)
    
    

def starter():
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 Testing File')
    parser.add_argument('--input_file', default='inputfile.txt', type=str, help='Path to input file')
    parser.add_argument('--output_file' , default = 'outputfile.txt' , type = str , help = 'Path to output files')
    parser.add_argument('--model_directory' , default = 'model', type=str , help = 'Path to saved models')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = starter()
    test(args)