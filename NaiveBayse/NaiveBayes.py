import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.pipeline import Pipeline
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
import pickle
from enum import Enum 
#import xgboost as xgb

## Parse args
parser = argparse.ArgumentParser("MultinomialNBTraining")
parser.add_argument("--Training_Data", type=str, help="Training dataset")
parser.add_argument("--Lable_Col", type=str, help="Lable column in the dataset.")
parser.add_argument("--Model_Type", type=str, default='MultinomialNB', help="Name of a certain Naive Bayes Model")
parser.add_argument("--Alpha", type=float,default=1.0, help="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).")
parser.add_argument("--Fit_prior", type=bool, default=True, help="Whether to learn class prior probabilities or not. If false, a uniform prior will be used.")
parser.add_argument("--Model_FileName", type=str, help="Name of the model file.")
parser.add_argument("--Model_Path", type=str, help="Path to store MultinomialNB model file in pickle format.")
args = parser.parse_args()

## Load data from DataFrameDirectory to Pandas DataFrame
training_df = load_data_frame_from_directory(args.Training_Data).data

# Create Pipelien to collect both transformation of text to vectorized and Naive Bayes model 
if args.Model_Type == 'MultinomialNB':
	pipe = Pipeline([('CountVectorizer', CountVectorizer()), ('TfidfTransformer', TfidfTransformer()), ('MultinomialNB', MultinomialNB(alpha=args.Alpha, fit_prior=args.Fit_prior, class_prior=None))])
else:
	pipe = Pipeline([('CountVectorizer', CountVectorizer()), ('TfidfTransformer', TfidfTransformer()), ('BernoulliNB', BernoulliNB(alpha=args.Alpha, fit_prior=args.Fit_prior, class_prior=None))])



## Prepare training data
training_df_features = training_df[[c for c in training_df.columns if c!=args.Lable_Col]]
training_df_lable = training_df[args.Lable_Col].squeeze()
result = pd.Series() if training_df_features.empty else training_df_features.iloc[:,0]


# Fiting Pipeline with training data 
pipe.fit(result, training_df_lable)



## Output model
os.makedirs(args.Model_Path, exist_ok=True)
f = open(args.Model_Path + "/" + args.Model_FileName, 'wb')
pickle.dump(pipe, f)
f.close()


