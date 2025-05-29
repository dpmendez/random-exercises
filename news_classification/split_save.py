# split and save train and test data
import pandas as pd
import numpy as np
import kagglehub
import os

from sklearn.model_selection import train_test_split

from helper import *


# download dataset
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

print("Path to dataset files:", path)

real_df = pd.read_csv(os.path.join(path, "True.csv"))
fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))

# give labels
real_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([real_df, fake_df], axis=0).reset_index(drop=True)

print('unique subjects: ', df["subject"].unique())

# custom preprocessing
print('\ncustom preprocessing')
df.loc[df['subject'] == 'politicsNews', 'subject'] = 'politics'
df.loc[df['subject'] == 'worldnews', 'subject'] = 'world'
df.loc[df['subject'] == 'Government News', 'subject'] = 'government'
df.loc[df['subject'] == 'US_News', 'subject'] = 'usa'
df.loc[df['subject'] == 'left-news', 'subject'] = 'left'
df.loc[df['subject'] == 'Middle-east', 'subject'] = 'middle-east'
df.loc[df['subject'] == 'News', 'subject'] = 'news'

df["clean_text"] = (df["title"] + " " + df["text"]).apply(clean_text)

X = df["clean_text"]
y = df["label"]
domain = df["subject"]

print('unique subjects: ', df["subject"].unique())
print('df dtypes: ', df.dtypes)

# split into train and test parts
X_train, X_test, y_train, y_test, domain_train, domain_test = train_test_split(X, y, domain, test_size=0.2, random_state=24, stratify=y)
print('\n\nsplit done')

# convert to dataframes
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train, name='label')
y_test = pd.Series(y_test, name='label')
domain_train = pd.Series(domain_train, name='domain')
domain_test = pd.Series(domain_test, name='domain')
print('df conversion done')

# concatenate features with labels and domains
train_df = pd.concat([X_train, y_train, domain_train], axis=1)
test_df = pd.concat([X_test, y_test, domain_test], axis=1)
print('concatenation done')

# save to csv
train_output = "train_data.csv"
test_output = "test_data.csv"
train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)
print('dataframes saved in ', train_output, ' and ', test_output)