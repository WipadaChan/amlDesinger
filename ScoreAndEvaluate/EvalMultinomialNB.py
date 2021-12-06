import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
import pickle
from sklearn import metrics
#import xgboost as xgb

## Parse args
parser = argparse.ArgumentParser("MultinomialNBEvaluation")
parser.add_argument("--Evaluation_Data", type=str, help="Evaluation dataset.")
parser.add_argument("--Lable_Col", type=str, default='None', help="Lable column in the evaluation dataset.")
parser.add_argument("--Action_Type", type=str, default='Score And Evaluate', help="Select action type")
parser.add_argument("--Model_Path", type=str, help="Path where contains model file.")
parser.add_argument("--Model_FileName", type=str, help="Name of the model file.")
parser.add_argument("--Evaluation_Output", type=str, help="Evaluation result")
args = parser.parse_args()

## Load data from DataFrameDirectory to Pandas DataFrame
evaluation_df = load_data_frame_from_directory(args.Evaluation_Data).data

## Prepare evaluation data
evaluation_df_features = evaluation_df[[c for c in evaluation_df.columns if c!=args.Lable_Col]]
result = pd.Series() if evaluation_df_features.empty else evaluation_df_features.iloc[:,0]

#vect = CountVectorizer()
#vect.fit(result)
#X_eval_dtm = vect.transform(result)

## Load model
os.makedirs(args.Model_Path, exist_ok=True)
f = open(args.Model_Path + "/" + args.Model_FileName, 'rb')
pipe= pickle.load(f)


if args.Action_Type == 'Score And Evaluate':
	## Evaluation
	evaluation_df_lable = evaluation_df[args.Lable_Col].squeeze()
	preds = pipe.predict(result)
	print("Accuracy Metric is ",metrics.accuracy_score(evaluation_df_lable, preds))
	print("Confusion Matrix: \n",metrics.confusion_matrix(evaluation_df_lable, preds))
	## Output evaluation result
	evaluation_df_features['Predict Result'] = pd.DataFrame(preds.T)
else: 
	preds = pipe.predict(result)
	evaluation_df_features['Predict Result'] = pd.DataFrame(preds.T)

os.makedirs(args.Evaluation_Output, exist_ok=True)
save_data_frame_to_directory(args.Evaluation_Output, evaluation_df_features)



