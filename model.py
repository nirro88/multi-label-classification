import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Other Libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
import ppscore as pps

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score, cross_val_predict

train_path = r'C:\Users\nirro\Desktop\machine learning\Earnix\train file exercise.csv'
df = pd.read_csv(train_path, dtype={"date_time": "string"})

# ----------------- get first information on data-set -----------------------------------
# information about the data dtype
# train_info = df.info()
# get statistical data on numeric columns
train_describe = df.describe()
# find how many null values in dataset
len1=len(df)
train_null = df.isnull().sum()
# each cleaning and prepossessing we will do on the train data-set i also will do on the test data-set


# ---------------------- imbalanced data ----------------------------------------------------------------------
# In this script, will use the following tactics for handling imbalanced classes:

# 1 - Up-sample the minority class
# 2 - Down-sample the majority class
# 3 - Change your performance metric
# 4 - Penalize algorithms (cost-sensitive training)
# 5 - Use tree-based algorithms an others

# after some analysis i discover that the data is imbalance, meaning one class in our case (0) is heavily skewed
percentage_class_0 = df.booking_bool.value_counts()[0]/len(df)*100
percentage_class_1 = df.booking_bool.value_counts()[1]/len(df)*100

# ----------------------------- plot tha class ---------------------------------------------------------------

# fig, ax =plt.subplots(1,2)
# sns.countplot(df['booking_bool'], ax=ax[0])
# sns.countplot(df['click_bool'], ax=ax[1])
# fig.show()

# ---------------------------- prepossessing and feature engineering ----------------------------------------------

# there is a lot of missing values, we need to deal with this first
# for start we drop all rows in which all the values in the row is null

# this action will drop about 11763 rows of empty data
df = df.dropna(how='all',axis=0)
# also we droped the empty columns, all the column values were empty
df = df.dropna(how='all', axis=1)



# fill the null values
list_columns_null_1 = ['comp2_rate','comp2_inv','comp3_rate','comp3_inv',
                      'comp5_rate','comp5_inv','comp8_rate','comp8_inv']
list_columns_null_2 = ['comp2_rate_percent_diff','comp3_rate_percent_diff',
                       'comp5_rate_percent_diff','comp8_rate_percent_diff']
for column in list_columns_null_1:
    df[column] = df[column].fillna(0)
for column in list_columns_null_2:
    df[column] = df[column].fillna(0)


# convert the date column from string to datetime
df['date_time'] = [pd.to_datetime(date) for date in df.date_time]
# add year month and day columns to df
df['year'] = [date.year for date in df.date_time]
df['month'] = [date.month for date in df.date_time]
df['day'] = [date.day for date in df.date_time]
df['hour'] = [date.hour for date in df.date_time]


# create target column - 'relevancy_score' where 0=not clicked at all, 1=clicked but not booked, 5=booked.
relevancy_score_list = []
for i in df.index:
    if df.click_bool[i] == 1 and df.booking_bool[i] == 1:
        relevancy_score_list.append(5)
    elif df.click_bool[i] == 1 and df.booking_bool[i] == 0:
        relevancy_score_list.append(1)
    elif df.click_bool[i] == 0 and df.booking_bool[i] == 0:
        relevancy_score_list.append(0)
df['relevancy_score'] = relevancy_score_list


# fill in the column - gross_bookings_usd in zeros
df['gross_bookings_usd'] = df['gross_bookings_usd'].fillna(0)
# fill in the column - prop_review_score
df['prop_review_score'] = df['prop_review_score'].fillna(df.prop_review_score.mean())


# columns name 'site_id' and 'visitor_location_country_id' contained only zero values, we will drop it.
df = df.drop(['site_id', 'visitor_location_country_id', 'prop_country_id'], axis=1)


# fill nans values in column 'prop_location_score2' with the mean value of that column
df.prop_location_score2 = df.prop_location_score2.fillna(df.prop_location_score2.mean())


# the below columns for now we decided to drop
# some had to much missing values some i didn't decided what to do soo i drop for now
df = df.drop(['visitor_hist_starrating','visitor_hist_adr_usd','srch_query_affinity_score','orig_destination_distance'], axis=1)

# check again for missing values
df_null = df.isnull().sum()

# ------------------------------------------------ Predictive Power Score and Correlation-------------------------

# matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
# sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", annot=True,xticklabels=True, yticklabels=True )
#
# plt.figure(figsize=(16,12))
# sns.heatmap(df.corr(),annot=True,fmt=".2f")

# ----------------------------------- Scaling and Distributing ------------------------------------------

# for now we drop this columns ['click_bool','booking_bool','gross_boking_usd','date_time']
df_for_scale = df.drop(['click_bool', 'booking_bool', 'gross_bookings_usd', 'date_time','comp4_inv'], axis=1)


# we will first scale the columns
scaled_x = df_for_scale.iloc[:,:-1]
rob_scaler = RobustScaler()
scaled_df = pd.DataFrame(rob_scaler.fit_transform(scaled_x),columns=df_for_scale.iloc[:,:-1].columns)
# scaled_test_df = rob_scaler.fit_transform(test)


# ---------------------------- split the train - data-set into train and validation -------------------------------

# we need to also create a sub sample of the dataframe in order to have an equal amount of booking and Non-booking cases.
# helping our algorithms better understand patterns that determines whether a search lead to booking or not.
# our subsample will be a data-frame with a 50/50 ratio of booking and and non-booking searches.
# we can choose different ratio.

y = df['relevancy_score']
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, validation_index in sss.split(scaled_df, y):
    # print("Train:", train_index, "Test:", validation_index)
    x_train, x_validation = scaled_df.iloc[train_index], scaled_df.iloc[validation_index]
    y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]


# Check the Distribution of the labels
# See if both the train and test label distribution are similarly distributed

# print(y_train.value_counts()[0]/len(y_train)*100)
# print(y_train.value_counts()[5]/len(y_train)*100)
# print(y_train.value_counts()[1]/len(y_train)*100)
# print(y_validation.value_counts()[0]/len(y_validation)*100)
# print(y_validation.value_counts()[5]/len(y_validation)*100)
# print(y_validation.value_counts()[1]/len(y_validation)*100)


# ------------------------------------------ Random Under and Over Sampling --------------------------------------------

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.
# ------------------------------ SMOTE - Synthetic Minority Oversampling Technique---------------------------------.
oversample = SMOTE()
oversample_x, oversample_y = oversample.fit_resample(x_train, y_train)

# summarize the new class distribution
# print(oversample_y.value_counts())


# The main issue with "Random Under-Sampling" is that we run the risk that our classification models
# will not perform as accurate as we would like to since there is a great deal of information loss
undersample = RandomUnderSampler()
undersample_x, undersample_y = undersample.fit_resample(x_train, y_train)

# summarize the new class distribution
# print(undersample_y.value_counts())


# check correlation and pps again
# oversample_x.insert(36,'y',oversample_y)
# matrix_df = pps.matrix(oversample_x)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
# sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True,xticklabels=True, yticklabels=True)
# oversample_x=oversample_x.drop('y', axis=1)

# after using pps test we can see that the most influencing columns:
# 'position','price_usd', 'day', 'hour','prop_location_score1','prop_log_historical_price',
# 'srch_boking_window','random bool' prop_starrating'

# it is important to notice that pps finds relation not correlation ,
# and every time you ran it it can basically product different results


# plt.figure(figsize=(16,12))
# oversample_x.insert(36,'y',oversample_y)
# sns.heatmap(oversample_x.corr(),annot=True,fmt=".2f",xticklabels=True, yticklabels=True)
# oversample_x=oversample_x.drop('y', axis=1)

# we can see that columns 'position' and 'random bool' has strongest correlation with the target



# # ------------------------------------ make predictions ----------------------------------------------
# # make predictions with over-sample

over_svm_classifier = SVC(decision_function_shape='ovr')
over_svm_classifier.fit(oversample_x, oversample_y)
over_svm_predictions = over_svm_classifier.predict(x_validation)

# over_xgb_classifier = OneVsRestClassifier(XGBClassifier())
# over_xgb_classifier.fit(oversample_x, oversample_y)
# over_xbg_predictions = over_xgb_classifier.predict(x_validation)
#
# over_rfc_classifier = OneVsRestClassifier(RandomForestClassifier())
# over_rfc_classifier.fit(oversample_x, oversample_y)
# over_rfc_predictions = over_rfc_classifier.predict(x_validation)
#
# over_knn_classifier = KNeighborsClassifier()
# over_knn_classifier.fit(oversample_x, oversample_y)
# over_knn_predictions = over_knn_classifier.predict(x_validation)
#
# # make predictions with under-sample
# under_svm_classifier = SVC(decision_function_shape='ovr')
# under_svm_classifier.fit(undersample_x, undersample_y)
# under_svm_predictions = under_svm_classifier.predict(x_validation)
#
# under_xgb_classifier = OneVsRestClassifier(XGBClassifier())
# under_xgb_classifier.fit(undersample_x, undersample_y)
# under_xbg_predictions = under_xgb_classifier.predict(x_validation)
#
# under_rfc_classifier = OneVsRestClassifier(RandomForestClassifier())
# under_rfc_classifier.fit(undersample_x, undersample_y)
# under_rfc_predictions = under_rfc_classifier.predict(x_validation)
#
# under_knn_classifier = KNeighborsClassifier()
# under_knn_classifier.fit(undersample_x, undersample_y)
# under_knn_predictions = under_knn_classifier.predict(x_validation)
# ------------------------------------------ clasification report --------------------------------------------
# Terms:
# True Positives: Correctly Classified Fraud Transactions
# False Positives: Incorrectly Classified Fraud Transactions
# True Negative: Correctly Classified Non-Fraud Transactions
# False Negative: Incorrectly Classified Non-Fraud Transactions
# Precision: True Positives/(True Positives + False Positives)
# Recall: True Positives/(True Positives + False Negatives)

# Precision as the name says, says how precise (how sure) is our model in detecting fraud transactions while recall is the amount of fraud cases our model is able to detect.
# Precision/Recall Tradeoff or F1 Score:
# The more precise (selective) our model is, the less cases it will detect.
# Example: Assuming that our model has a precision of 95%, Let's say there are only 5 fraud cases in which the model is 95% precise or more that these are fraud cases.
# Then let's say there are 5 more cases that our model considers 90% to be a fraud case, if we lower the precision there are more cases that our model will be able to detect.
#
#
# # classification_report for over-sample data
over_svm_report = classification_report(y_validation, over_svm_predictions)

# over_xbg_report = classification_report(y_validation, over_xbg_predictions)
# over_rfc_report = classification_report(y_validation, over_rfc_predictions)
# over_knn_report = classification_report(y_validation, over_knn_predictions)
# a=np.array([over_knn_predictions,over_svm_predictions,over_rfc_predictions,over_xbg_predictions,y_validation]).T
# over_result_df = pd.DataFrame(data=a,columns=['knn','svm','random_forest','xbg','y_validation'])

# # classification_report for under-sample data
# under_svm_report = classification_report(y_validation, under_svm_predictions)
# under_xbg_report = classification_report(y_validation, under_xbg_predictions)
# under_rfc_report = classification_report(y_validation, under_rfc_predictions)
# under_knn_report = classification_report(y_validation, under_knn_predictions)
# a2=np.array([under_knn_predictions,under_svm_predictions,under_rfc_predictions,under_xbg_predictions,y_validation]).T
# under_result_df = pd.DataFrame(data=a2,columns=['knn','svm','random_forest','xbg','y_validation'])
#
#

# # -------------------------- now we will make predictions on the test data ---------------------------------------
# # in order to make prediction on the test data we first have to preprocessed the same way like the train datd
#
# # # ----------------------- preprocessed the test data-set before make prediction -------------------------------

test_path = r'C:\Users\nirro\Desktop\machine learning\Earnix\test file exercise.csv'
df_test = pd.read_csv(test_path, dtype={"date_time": "string"})

# ----------------- get first information on data-set -----------------------------------
# information about the data dtype
df_test_info = df_test.info()
# get statistical data on numeric columns
df_test_describe = df_test.describe()
# find how many null values in dataset
len2=len(df_test)
df_test_null = df_test.isnull().sum()
# each cleaning and prepossessing we will do on the train data-set i also will do on the test data-set

# ---------------------------- prepossessing and feature engineering ----------------------------------------------

# there is a lot of missing values, we need to deal with this first
# for start we drop all rows in which all the values in the row is null

# this action will drop about 11763 rows of empty data
df_test = df_test.dropna(how='all',axis=0)
# also we droped the empty columns, all the column values were empty
df_test = df_test.dropna(how='all', axis=1)



# fill the null values
list_columns_null_1 = ['comp2_rate','comp2_inv','comp3_rate','comp3_inv',
                       'comp5_rate','comp5_inv','comp8_rate','comp8_inv']
list_columns_null_2 = ['comp2_rate_percent_diff','comp3_rate_percent_diff',
                       'comp5_rate_percent_diff','comp8_rate_percent_diff']
for column in list_columns_null_1:
    df_test[column] = df_test[column].fillna(0)
for column in list_columns_null_2:
    df_test[column] = df_test[column].fillna(0)


# convert the date column from string to datetime
df_test['date_time'] = [pd.to_datetime(date) for date in df_test.date_time]
# add year month and day columns to df
df_test['year'] = [date.year for date in df_test.date_time]
df_test['month'] = [date.month for date in df_test.date_time]
df_test['day'] = [date.day for date in df_test.date_time]
df_test['hour'] = [date.hour for date in df_test.date_time]

# fill in the column - prop_review_score
df_test['prop_review_score'] = df_test['prop_review_score'].fillna(df_test.prop_review_score.mean())


# columns name 'site_id' and 'visitor_location_country_id' contained only zero values, we will drop it.
df_test = df_test.drop(['site_id', 'visitor_location_country_id', 'prop_country_id'], axis=1)


# fill nans values in column 'prop_location_score2' with the mean value of that column
df_test.prop_location_score2 = df_test.prop_location_score2.fillna(df_test.prop_location_score2.mean())


# the below columns for now we decided to drop
# some had to much missing values some i didn't decided what to do soo i drop for now
df_test = df_test.drop(['visitor_hist_starrating','visitor_hist_adr_usd','srch_query_affinity_score','orig_destination_distance','date_time'], axis=1)

# check again for missing values
df_test_null = df_test.isnull().sum()


# ----------------------------------- Scaling and Distributing ------------------------------------------

# for now we drop this columns ['click_bool','booking_bool','gross_boking_usd','date_time']
# df_test_for_scale = df_test.drop(['click_bool', 'booking_bool', 'gross_bookings_usd', 'date_time'], axis=1)

rob_scaler = RobustScaler()
test_scaled_df = rob_scaler.fit_transform(df_test)


# # ----------------------------------------- make predictions ---------------------------------------------------

test_predictions = over_svm_classifier.predict(test_scaled_df)
# over_svm_report = classification_report(y, over_svm_predictions)


# ----------------------------------- suggestion --------------------------------------------------------

# preprocessed the test data-set before make prediction
# Cost-Sensitive Learning -
# Algorithms can be modified to change the way learning is performed to bias towards those classes that have
# fewer examples in the training dataset.
# example:
# weights = {0:1.0, 1:2.0, 5:3.0}
# model = RandomForestClassifier(n_estimators=1000, class_weight=weights)

# use stacking model -
# combine several classification models and then make prediction throw voting

# clustring -
# split the original dataset to three small data set for each class,
# by clustering algorithms or by other method, and the make assumptions on the different groups
#
#
