# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:42:31 2024

@author: priya
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from urllib.parse import quote
import sweetviz as sv
import sidetable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import joblib
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skmet
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Importing the data using sql

machine = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Machine Downtime.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = 'priya'  # password
db = 'mach'  # database name
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote (f'{pw}'))

# to_sql() - function to push the dataframe onto a SQL table.

machine.to_sql('mach1_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from mach1_tbl;'
df = pd.read_sql_query(sql, engine)


######################   DATA UNDERSTANDING     ############################

duplicates = df.duplicated()

# Print the number of duplicate rows
print(duplicates.sum())

# check null values if any
null_values = df.isnull()
null_count = null_values.sum()

print(null_count)
# There are null values present 
# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the DataFrame
print(df)

df.describe()

# dropping Dates column
df.drop(columns=['Dates'], inplace=True)

# We have to check unique values for categorical data 
df.Downtime.value_counts()
# There were 1265 instances of machine failure and 1235 instances of no machine failure.

df.Machine_ID.value_counts()
# Makino-L1-Unit1-2013: This value appears 874 times in the "Machine_ID" column.
# Makino-L3-Unit1-2015: This value appears 818 times in the "Machine_ID" column. 
# Makino-L2-Unit1-2015: This value appears 808 times in the "Machine_ID" column.

df.Assembly_Line_No.value_counts()
# Shopfloor-L1: This value appears 874 times in the "Assembly_Line_No" column.
# Shopfloor-L3: This value appears 818 times in the "Assembly_Line_No" column.
# Shopfloor-L2: This value appears 808 times in the "Assembly_Line_No" column.

###################### AUTO EDA ####################
#EDA using Autoviz
sweet_report = sv.analyze(df)

#Saving results to HTML file
sweet_report.show_html('sweet_report.html')



########################  EXPLORATORY DATA ANALYSIS / DESCRIPTIVE STATISTICS   ###########################

# FIRST MOMENT BUSINESS DECISION /MEASURE OF CENTRAL TENDENCY

# Mode for coulmns  which is categorical 
mode_value = df[['Machine_ID','Assembly_Line_No','Downtime']].mode()
print(mode_value)
# MODE VALUES - Machine_ID: Makino-L1-Unit1-2013  , Assembly_Line_No: Shopfloor-L1  and Downtime: Machine_Failure

# Calculate mean and median for each column
mean_values = df.mean()
median_values = df.median()

# Print mean and median values
print("Mean values:")
print(mean_values)
print("\nMedian values:")
print(median_values)

#  SECOND MOMENT BUSINESS DECISION / MEASURE OF DISPERSION 

# Calculate standard deviation, range, and variance for each column

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])


std_deviation_values = numeric_columns.std()
range_values = df.max() - numeric_columns.min()
variance_values = numeric_columns.var()

# Print standard deviation, range, and variance values
print("Standard Deviation:")
print(std_deviation_values)
print("\nRange:")
print(range_values)
print("\nVariance:")
print(variance_values)


# THIRD MOMENT BUSINESS DECISION / SKEWENESS

# Calculate skewness for each column
skewness_values = df.skew()

# Print skewness  values
print("Skewness:")
print(skewness_values)


# FOURTH MOMENT BUSINESS DECISION /KURTOSIS

# Calculate kurtosis for each column
kurtosis_values = df.kurtosis()

# Print  kurtosis values
print("\nKurtosis:")
print(kurtosis_values)



##################   DATA CLEANING ########################

# Input and Output Split

predictors = df.loc[:, df.columns != "Downtime"]
type(predictors)

predictors.columns

target = df["Downtime"]
type(target)

target

# Checking for duplicate values if any
df.duplicated().sum()
# There are no duplicate values


# check null values if any
null_values = df.isnull()
null_count = null_values.sum()

print(null_count)
# There are null values present 


# **By using Mean imputation null values can be impute**
numeric_features = df.select_dtypes(exclude = ['object']).columns
numeric_features

# Non-numeric columns
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('Downtime')


################  Missing values Analysis        ###############################

# Checking for Null values
df.isnull().sum()
## There are null values present.


# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

imputation = preprocessor.fit(predictors)

## Save the pipeline
joblib.dump(imputation, 'meanimpute')

cleandata = pd.DataFrame(imputation.transform(predictors), columns = numeric_features)
cleandata

cleandata.isnull().sum()
# all missing values have been imputed successfully in your data

##########  Scaling with MinMaxScaler      #################################

scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata)

joblib.dump(scale, 'minmax')

scaled_data = pd.DataFrame(scale.transform(cleandata), columns=cleandata.columns)
scaled_data
# all values are scaled using minmaxscaler successfully in your data

# Plot histograms for each column in scaled data after scaling
plt.figure(figsize=(20, 10))
for i, col in enumerate(scaled_data.columns, 1):
    plt.subplot(5, 3, i)  # Adjust the subplot grid as per your number of columns
    plt.hist(scaled_data[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot histograms for each column before scaling
plt.figure(figsize=(20, 10))
for i, col in enumerate(scaled_data.columns, 1):
    plt.subplot(5, 3, i)  # Adjust the subplot grid as per your number of columns
    plt.hist(predictors[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# here ,we can notice that after scaling the x -axis values are changed to 0 to 1 , means all numerical values are now scaled

###########  Encoding Non-numeric fields         #################################

# **Convert Categorical data  to Numerical data using OneHotEncoder**

 # Define the encoding pipeline for one-hot encoding
# encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

# encoding_columntransfer = ColumnTransformer([('encode', encoding_pipeline, categorical_features)])
# encoding_pipeline = encoding_columntransfer.fit(predictors)

# joblib.dump(encoding_pipeline, 'encoding')

# encode_data = pd.DataFrame(encoding_pipeline.transform(predictors))
# encode_data

# # clean_data = pd.concat([scaled_data], axis = 1, ignore_index = True)  # concatenated data will have new sequential index
# # clean_data


# File gets saved under current working directory
import os
os.getcwd()

scaled_data.describe()
scaled_data.columns
scaled_data.info()
scaled_data.head


############## Outlier Analysis        #######################

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

scaled_data.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#### Outlier analysis: Columns are continuous, hence outliers are treated

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
                                 'Air_System_Pressure(bar)', 'Coolant_Temperature',
                                 'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
                                 'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
                                 'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)'])


outlier = winsor.fit(scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
       'Air_System_Pressure(bar)', 'Coolant_Temperature',
       'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
       'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
       'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']])

# Save the winsorizer model '
joblib.dump(outlier, 'winsor')

scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
       'Air_System_Pressure(bar)', 'Coolant_Temperature',
       'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
       'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
       'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']] = outlier.transform(scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
              'Air_System_Pressure(bar)', 'Coolant_Temperature',
              'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
              'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
              'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']])

scaled_data
# Re-checking if all outliers are treated
scaled_data.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# All outliers have been treated, missing values have been imputed, categorical data has been encoded, and the dataset is now prepared for model building.
