## Starbucks Capstone Project

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

## Medium Blog Link
A detailed report of analysis for this project is available [here](https://medium.com/@jaskaranbhatia/starbucks-capstone-project-predicting-offer-effectiveness-b09174056f9)

## Libraries Used
 
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings

## Motivation

This is the project to complete Udacity Data Scientist Nanodegree Capstone Project 
I have chosen Starbucks data that mimics customer behavior on the Starbucks rewards mobile app and build a model to predict the response of customers to a particular offer

## Datasets and Inputs

For this project, the data sets are provided by Starbucks and Udacity in the form of three JSON files. These contains simulated data that mimics customer behavior on the Starbucks rewards mobile app.
-   portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
-   profile.json - demographic data for each customer
-   transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**

-   id (string) - offer id
-   offer_type (string) - type of offer ie BOGO, discount, informational
-   difficulty (int) - minimum required spend to complete an offer
-   reward (int) - reward given for completing an offer
-   duration (int) - time for offer to be open, in days
-   channels (list of strings)

**profile.json**

-   age (int) - age of the customer
-   became_member_on (int) - date when customer created an app account
-   gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
-   id (str) - customer id
-   income (float) - customer's income

**transcript.json**

-   event (str) - record description (ie transaction, offer received, offer viewed, etc.)
-   person (str) - customer id
-   time (int) - time in hours since start of test. The data begins at time t=0
-   value - (dict of strings) - either an offer id or transaction amount depending on the record

## Files
Starbucks_Capstone_notebook.ipynb: the code notebook<br/>
Starbucks_Capstone_notebook.html : the HTML version of the notebook<br/>
report.pdf : the report explaining all the stages of capstone project<br/>
links.txt : contains link to Medium Blog And Github Repo

## Summary 

1) the best score is by DecisionTreeClassifier model, as its test F1 score is 85.1

2) the lowest of all models here was KNeighborsClassifier model by with test F1 score of 32.9

3) the company should focus to give more interesting offers to males since their income is higher

4) Discount and BOGO increase the customer buy rating

## Acknowledgements
Special thanks to Starbucks and Udacity for providing the data utilized in this project
