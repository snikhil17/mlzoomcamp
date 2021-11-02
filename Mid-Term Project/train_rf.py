# -*- coding: utf-8 -*-
"""
# **Problem Statement:**

- The major objective of this project is to extract actionable insights from the historical match data and make strategic changes to make India win. 
- Primary objective is to create Machine Learning models which correctly predicts a win for the Indian Cricket Team. 
- Once a model is developed then you have to extract actionable insights and recommendation. Also, below are the details of the next 10 matches, India is going to play. You have to predict the result of the matches.

## **Consider following matches and predict the result:**

1. Test match with England in England. All the match are day matches. In England, it will be rainy season at the time to match. 

2. T20 match with Australia in India. All the match are Day and Night matches. In India, it will be winter season at the time to match. 

3. ODI match with Sri Lanka in India. All the match are Day and Night matches. In India, it will be winter season at the time to match.
"""

# Loading Libraries
# !pip install optuna
import warnings
# import requests
import pickle
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
print("Loading Libraries...")
# import optuna


plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use("fivethirtyeight")


warnings.filterwarnings('ignore')


# parameters
params = {'n_estimators': 3925, 'max_depth': int(31.908468045247087)}

"""Save the model"""
output_file = f'model_rf.bin'

print("Loading Data")
# Loading Data
df_maindata = pd.read_excel(
    r'C:\Users\nshre\Desktop\mid -term project\Sports Data.xlsx', sheet_name='Sports data for DSBA')
df_maindata.info()

"""Making columns lower case and replacing any spaces with '_'"""
df_maindata.columns = df_maindata.columns.str.lower().str.replace(' ', '_')

percentage = df_maindata.isnull().mean().round(5).to_frame().rename(
    {0: '%age of Missing Values'}, axis=1).sort_values(by='%age of Missing Values', ascending=False)
plot_percentage = percentage.reset_index().rename(
    {"index": "Variables"}, axis=1)

missing_values_cols = list(
    plot_percentage[plot_percentage['%age of Missing Values'] != 0]['Variables'])
missing_cat_cols = [
    col for col in missing_values_cols if df_maindata[col].dtype == 'object']
missing_num_cols = [
    col for col in missing_values_cols if col not in missing_cat_cols]

"""Drop rows with missing values in Opponent"""
df_maindata.dropna(subset=['opponent'], inplace=True)

# Now removing Opponent col from list of categorical colummns
missing_cat_cols.remove('opponent')

"""Impute Mode for categorical columns"""
for col in missing_cat_cols:
    df_maindata[col].fillna(value=df_maindata[col].mode()[0], inplace=True)

"""Impute mean for numerical columns"""
for col in missing_num_cols:
    df_maindata[col].fillna(value=df_maindata[col].median(), inplace=True)

"""Replacing repeated values"""

df_maindata['player_highest_wicket'] = df_maindata['player_highest_wicket'].apply(
    lambda x: x if (x != 'Three') else 3)
df_maindata['players_scored_zero'] = df_maindata['players_scored_zero'].apply(
    lambda x: x if (x != 'Three') else 3)
df_maindata['match_format'] = df_maindata['match_format'].apply(
    lambda x: x if (x != '20-20') else 'T20')
df_maindata['first_selection'] = df_maindata['first_selection'].apply(
    lambda x: x if (x != 'Bat') else 'Batting')


"""Converting player_highest_wicket and players_scored_zero to integer"""
df_maindata['player_highest_wicket'] = df_maindata['player_highest_wicket'].astype(
    'int')
df_maindata['players_scored_zero'] = df_maindata['players_scored_zero'].astype(
    'int')

# Dropping the constant feature and Game number column
df_maindata.drop(['wicket_keeper_in_team', 'game_number'],
                 axis=1, inplace=True)

# Removing multi-collinear variables
df_maindata.drop(['audience_number', 'extra_bowls_bowled'],
                 axis=1, inplace=True)

# Treating Outliers
percentiles = df_maindata['avg_team_age'].quantile([0.01, 0.99]).values
df_maindata['avg_team_age'] = np.clip(
    df_maindata['avg_team_age'], percentiles[0], percentiles[1])

# Converting Result Variable into Binary form
df_maindata['result'] = df_maindata['result'].apply(
    lambda x: 1 if x == 'Win' else 0)


useful_cols = [col for col in df_maindata.columns if col not in [
    'game_number', 'result', 'kfold']]
categorical = [
    col for col in useful_cols if df_maindata[col].dtype == 'object']
numerical = [col for col in useful_cols if col not in categorical]

df_train, df_test1 = train_test_split(
    df_maindata, stratify=df_maindata['result'], test_size=0.15, random_state=7)


def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # params = {'colsample_bytree': 0.6123502038809282,
    #           'gamma': 0.0008327772926904814,
    #           'learning_rate': 0.0592982016039167,
    #           'max_depth': 6,
    #           'n_estimators': 8819,
    #           'subsample': 0.7841121520382168}
    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier(**params, random_state=7)
    model_rf.fit(X_train, y_train)

    return dv, model_rf


def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Preparing data as a tabular matrix
y = df_train.result
X = df_train.drop('result', axis=1)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

scores = []

for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):

    df_tr = df_train.iloc[train_idx]
    df_val = df_train.iloc[val_idx]

    y_train = df_tr.result.values
    y_val = df_val.result.values

    X_train = df_tr[useful_cols]
    X_val = df_val[useful_cols]

    # dv, model = train(X_train, y_train)
    # y_pred = predict(X_val, dv, model)
    train_dicts = X_train.to_dict(orient='records')
    val_dicts = X_val.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    # params = study.best_params
    model_rf = RandomForestClassifier(**params, random_state=7)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict_proba(X_val)[:, 1]
    roc_auc = metrics.roc_auc_score(y_val, y_pred)

    scores.append(roc_auc)
    print(fold, roc_auc)

print(' %s +- %s' % (np.mean(scores), np.std(scores)))

df_test = df_test1[useful_cols].copy()
dv, model = train(X, y)
y_pred = predict(df_test, dv, model)

auc = metrics.roc_auc_score(df_test1.result.values, y_pred)
print(f'auc={auc}')
# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


print(f'the model is saved to {output_file}')


# """Making requests"""


# url = 'http://localhost:9696/predict'

# match = {'all_rounder_in_team': 1.0,
#          'avg_team_age': 26.0,
#          'bowlers_in_team': 2.0,
#          'extra_bowls_opponent': 0,
#          'first_selection': 'Bowling',
#          'match_format': 'T20',
#          'match_light_type': 'Day',
#          'max_run_given_1over': 6.0,
#          'max_run_scored_1over': 12.0,
#          'max_wicket_taken_1over': 1,
#          'min_run_given_1over': 0,
#          'min_run_scored_1over': 3.0,
#          'offshore': 'No',
#          'opponent': 'South Africa',
#          'player_highest_run': 45.0,
#          'player_highest_wicket': 2,
#          'players_scored_zero': 3,
#          'season': 'Winter'}

# response = requests.post(url, json=match).json()

# response

# if response['churn']:
#     print('sending email to', 'asdx-123d')
