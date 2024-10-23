'''  Name     : METHRE PRIYA 
     Batch ID : 121323204012 '''

'**    PROJECT - OPTIMIZATION OF MACHINE DOWNTIME - DATA PREPROCESSING CODE       **'
        
'''
Business Problem    : Machines which manufacture the pumps. Unplanned Machine Downtime which
is leading to loss of productivity.

Business Objective  : Minimize unplanned machine downtime.
Business Constraint : Minimize maintenance cost.

Success Criteria:
Business Success Criteria         : Reduce the unplanned downtime by at least 10%.
Machine Learning Success Criteria :  Develop an ML model that reduces unplanned machine downtime by at least 10%.
Economic Success Criteria         : Achieve a cost saving of at least $1M.

Data Dictionary:
 Dataset contains 2500 entries
 15 features are recorded 

 Description:
1. Machine ID  -	an identifier for each machine in the dataset.
2. Assembly Line Number  -	Indicates the assembly line to which the machine belongs.
3. Hydraulic Pressure (bar) - The pressure of the hydraulic system in bars. Hydraulic pressure is critical for the proper functioning of many industrial machines.
4. Coolant Pressure (bar) -	Pressure of the coolant system in bars. Coolant is often used to regulate the temperature of machines during operation.
5. Air System Pressure (bar) - Pressure of the air system in bars. This could be relevant for pneumatic systems in the machinery.
6. Coolant Temperature - Temperature of the coolant in the machine, usually measured in degrees Celsius.
7. Hydraulic Oil Temperature (°C) -	Temperature of the hydraulic oil in the machine, measured in degrees Celsius.
8. Spindle Bearing Temperature (°C)	- Temperature of the spindle bearings, which are crucial components in many machining processes, measured in degrees Celsius.
9. Spindle Vibration (µm) - Vibration level of the spindle, which can indicate the stability and performance of the machining process, measured in micrometers (µm).
10. Tool Vibration (µm) - Vibration level of the tool used in the machining process, measured in micrometers (µm).
11. Spindle Speed (RPM)	 - Rotations per minute of the spindle, which determines the cutting speed in machining operations.
12. Voltage (volts)	-  Electrical voltage supplied to the machine, measured in volts.
13. Torque (Nm) - Torque applied to the machine, measured in Newton-meters (Nm). Torque is crucial for the operation of rotating machinery.
14. Cutting (kN)- Cutting force applied during machining, measured in kilonewtons (kN). Cutting force is essential for understanding the load on the machine during operation.
15. Downtime - Duration of machine downtime, typically measured in hours or minutes. Downtime represents the period during which the machine is not operational, often due to maintenance, repairs, or failures.



'''

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

#################      GRAPHICAL REPRESENTATION      ########################

df.info()

sns.pairplot(df)   # original data

# Correlation Analysis on Original Data
orig_df_cor = df.corr()
orig_df_cor
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
# Heatmap

sns.heatmap(orig_df_cor,cmap ='twilight', xticklabels=orig_df_cor, yticklabels=orig_df_cor)
plt.title('Correlation Heatmap')
plt.show()


# Pie Chart

plt.figure(figsize=(8, 8))
df['Downtime'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Downtime Categories')
plt.ylabel('')
plt.show()
# It seems like there's a slightly higher occurrence of machine failure, accounting for 50.6%, compared to non-machine failures at 49.4%.

# Stacked Bar Plot with Assembly_Line_No
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Assembly_Line_No', hue='Downtime')
plt.title('Downtime Distribution Across Assembly Lines')
plt.xlabel('Assembly Line')
plt.ylabel('Frequency')
plt.legend(title='Downtime Category')
plt.show()
# Shopfloor-L1 exhibits the highest incidence of machine failure, while Shopfloor-L2 and Shopfloor-L3 demonstrate nearly equal rates of both machine failure and non-machine failure.

# Stacked Bar Plot with Machine ID

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Machine_ID', hue='Downtime')
plt.title('Downtime Distribution Across Machine_ID ')
plt.xlabel('Machine_ID')
plt.ylabel('Frequency')
plt.legend(title='Downtime Category')
plt.show()
# Makino-L1-Unit1-2013 shows the highest machine failure rate, while Makino-L2-Unit1-2015 and Makino-L3-Unit1-2015 display nearly identical rates of both machine failure and non-machine failure.

df.columns

# Bar plots between numerical values and Downtime

'  Hydraulic pressure VS Downtime' 

colors = ['yellow', '#9467bd']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Hydraulic_Pressure(bar)', palette=colors)
plt.title('Hydraulic Pressure by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Hydraulic Pressure (bar)')
plt.show()
# As hydraulic pressure increases, there tends to be a decrease in machine failure occurrences.


'  Coolant pressure VS Downtime     '

colors = ['maroon', 'black']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Coolant_Pressure(bar)', palette=colors)
plt.title('Coolant Pressure by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Coolant Pressure (bar)')
plt.show()
# The rise in coolant pressure doesn't substantially affect machine downtime.

'  Air system pressure VS Downtime  '

colors = ['#1f77b4', '#e377c2']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Air_System_Pressure(bar)', palette=colors)
plt.title('Air_System_Pressure(bar) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Air_System_Pressure(bar)')
plt.show()
# As the air system pressure increases, both machine failure and no-machine failure instances appear to rise proportionally, suggesting an equivalence between the two.

'  Coolant_Temperature VS Downtime   '

colors = ['#8c564b', '#17becf']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Coolant_Temperature', palette=colors)
plt.title('Coolant_Temperature by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Coolant_Temperature')
plt.show()
# As the temperature of the coolant increases, there is a corresponding uptick in instances of machine failure.

'  Hydraulic_Oil_Temperature(°C) VS Downtime '

colors = ['#f7b6d2', '#7f7f7f']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Hydraulic_Oil_Temperature(°C)', palette=colors)
plt.title('Hydraulic_Oil_Temperature(°C) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Hydraulic_Oil_Temperature(°C)')
plt.show()
# As hydraulic oil temperature increases, both machine and no-machine failure rates increase, reaching a comparable level.

' Spindle_Bearing_Temperature(°C) VS Downtime   '

colors = ['#c49c94','#ff9896']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Spindle_Bearing_Temperature(°C)', palette=colors)
plt.title('Spindle_Bearing_Temperature(°C) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Spindle_Bearing_Temperature(°C)')
plt.show()
#  As Spindle bearing temperature increases, both machine and no-machine failure rates increase, reaching a comparable level.

'  Spindle_Vibration(µm) VS Downtime   '

colors = ['#98df8a', '#aec7e8']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Spindle_Vibration(µm)', palette=colors)
plt.title('Spindle_Vibration(µm) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Spindle_Vibration(µm)')
plt.show()
# Spindle vibration exhibits an equal occurrence of both machine failure and non-machine failure instances.

' Tool_Vibration(µm) VS Downtime'

colors = ['#b35900', '#006600']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Tool_Vibration(µm)', palette=colors)
plt.title('Tool_Vibration(µm) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Tool_Vibration(µm)')
plt.show()
# Tool vibration exhibits an equal occurrence of both machine failure and non-machine failure instances.

'Spindle_Speed(RPM) VS Downtime'

colors = ['#4d004d', '#999900']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Spindle_Speed(RPM)', palette=colors)
plt.title('Spindle_Speed(RPM) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Spindle_Speed(RPM)')
plt.show()
# With an increase in spindle speed, there is a slight uptick observed in machine failure occurrences.

'Voltage(volts) VS Downtime'

colors = ['#c5b0d5', '#4d004d']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Voltage(volts)', palette=colors)
plt.title('Voltage(volts) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Voltage(volts)')
plt.show()
# Voltage exhibits an equal occurrence of both machine failure and non-machine failure instances.

'Torque(Nm) VS Downtime'

colors = ['#008080', '#708090']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Torque(Nm)', palette=colors)
plt.title('Torque(Nm) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Torque(Nm)')
plt.show()
# An increase in torque corresponds to a decrease in machine failure rates.

'Cutting(kN) VS Downtime'

colors = ['#40e0d0', '#fa8072']  
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Downtime', y='Cutting(kN)', palette=colors)
plt.title('Cutting(kN) by Downtime')
plt.xlabel('Downtime')
plt.ylabel('Cutting(kN)')
plt.show()
# As cutting force (measured in KN) rises, there is typically a corresponding increase in machine failure rates, coupled with a slight decrease in non-machine failure instances.

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Spindle_Speed(RPM)', y='Downtime', data=df, hue='Downtime')
plt.title('Scatter Plot of Spindle Speed vs. Downtime')
plt.xlabel('Spindle Speed (RPM)')
plt.ylabel('Downtime')
plt.legend(title='Downtime')
plt.grid(True)
plt.show()
scaled_data.info()


# Histogram for all features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with all features

# Calculate the number of features and define the layout of subplots
num_features = len(df.columns)
num_cols = 3  # Number of columns for subplots
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate number of rows

# Define the size of the figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# Flatten axes for easy iteration
axes = axes.flatten()

# Loop through each feature and create histograms
for i, col in enumerate(df.columns):
    ax = axes[i]
    ax.hist(df[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    ax.set_axisbelow(True)

# Hide any extra subplots
for i in range(num_features, len(axes)):
    axes[i].axis('off')

# Adjust layout and show plot
plt.tight_layout()
plt.show()




# Converting target variable to binary data using mapping function
target.info()

# Map values to binary using a dictionary
binary_map = {'Machine_Failure': 0, 'No_Machine_Failure': 1}
target = target.to_frame(name='Downtime')
target['Downtime'] = target['Downtime'].map(binary_map)

# Display the DataFrame to verify the changes
print(target)


############  MODEL BUILDING  ###################

scaled_data.columns

#2.--Feature Selection using Random Forest 
from sklearn.ensemble import RandomForestRegressor

#Initialize Random Forest Regressor 
rf = RandomForestRegressor()

#Fit the model to your preprocessed data 
rf.fit(scaled_data, target)

#Get feature importances
feature_importances = pd.DataFrame(rf.feature_importances_,index=scaled_data.columns,columns=['importance'])

#Sort the features by their importance
feature_important = feature_importances.sort_values(by='importance', ascending=False)

#Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances.index, feature_importances ['importance'], color='skyblue')
plt.xlabel('Importance') 
plt.ylabel('Features')
plt.title('Feature Importance') 
plt.show()
#Setting a threshold value (adjust this based on your preference)
threshold = 0.01

# the important features are are 'Hydraulic_Pressure(bar)','Coolant_Pressure(bar)','Torque(Nm)', 'Cutting(kN)' and 'Spindle_Speed(RPM)'

# Separate features (X) and target variable (y)
# We've identified the crucial features impacting downtime for the given X value using Randomforest analysis.
X = scaled_data.drop(columns=['Air_System_Pressure(bar)', 'Coolant_Temperature',
'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
'Spindle_Vibration(µm)', 'Tool_Vibration(µm)','Voltage(volts)'])
X.columns

Y = target
Y

# Check the shape of X and y
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

#  Assuming Y is your target variable ,converting it into 1D array
Y = np.ravel(Y)

# Data Partition into Train and Test
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, stratify = Y,random_state=42)

############# Random Forest Model


rf_Model = RandomForestClassifier()

# #### Hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2', None]

# Maximum number of levels in tree
max_depth = [2, 4]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree
bootstrap = [True, False]


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
n_estimators

# Create the param grid

param_grid = {'n_estimators': n_estimators,
               'max_features': ['auto', 'sqrt', 'log2', None],
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)

# ### Hyperparameter optimization with GridSearchCV

rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 10, verbose = 1, n_jobs = -1)

rf_Grid.fit(train_X,train_Y)

rf_Grid.best_params_

cv_rf_grid = rf_Grid.best_estimator_

# ## Check Accuracy

# Evaluation on Test Data

test_pred = cv_rf_grid.predict(test_X)

accuracy_test = np.mean(test_pred == test_Y)
accuracy_test

cm = skmet.confusion_matrix(test_Y, test_pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['test_Y', 'test_pred'])
cmplot.plot()
cmplot.ax_.set(title = ' Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

print (f'Train Accuracy - : {rf_Grid.score(train_X, train_Y):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(test_X, test_Y):.3f}')
#  The training accuracy score is 0.963%, suggesting that the model performs well on the training data.
#  The test accuracy score is 0.966%, indicating that the model generalizes well to unseen data, which is a positive sign of its effectiveness.
#  These accuracy scores suggest that the Random Forest model trained using GridSearchCV performs well on both the training and test datasets, with relatively high accuracy.

pickle.dump(cv_rf_grid, open('rfc.pkl', 'wb'))


############ DECISION TREE


dt = DecisionTreeClassifier(random_state=42)
dt_param_grid = {'max_depth': [None, 5, 10, 20]}
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=5, n_jobs=-1)
dt_grid_search.fit(train_X, train_Y)
dt_best_model = dt_grid_search.best_estimator_
dt_y_pred = dt_best_model.predict(test_X)
dt_accuracy = accuracy_score(test_Y, dt_y_pred)

# Calculate accuracy on the training set
dt_y_train_pred = dt_best_model.predict(train_X)
dt_train_accuracy = accuracy_score(train_Y, dt_y_train_pred)
dt_train_accuracy
# It seems like the Decision Tree Classifier has achieved a perfect accuracy of 0.9975 (or 100%) on the training data, indicating that it perfectly predicts the labels for the data it was trained on.

# Calculate accuracy on the test set
dt_y_pred = dt_best_model.predict(test_X)
dt_accuracy = accuracy_score(test_Y, dt_y_pred)
dt_accuracy
# The accuracy of 0.98% indicates that the Decision Tree Classifier is performing well in predicting the target variable (downtime) on unseen data.

# Saving Decision Tree Classifier
pickle.dump(dt_best_model, open('decision_tree_model.pkl','wb'))


#########  KNN

knn = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 10, 20]}
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, n_jobs=-1)
knn_grid_search.fit(train_X, train_Y)
knn_best_model = knn_grid_search.best_estimator_

# Predict on the training set
knn_y_train_pred = knn_best_model.predict(train_X)
knn_train_accuracy = accuracy_score(train_Y, knn_y_train_pred)
print("KNN accuracy on train data:", knn_train_accuracy)

# Predict on the test set
knn_y_pred =  knn_best_model.predict(test_X)
knn_accuracy = accuracy_score(test_Y, knn_y_pred)
knn_accuracy
# The K-Nearest Neighbors (KNN) Classifier achieved an accuracy of approximately 98.95% on the training data and approximately 96.6% on the test data.

# Saving K-Nearest Neighbors Classifier
pickle.dump(knn_best_model, open('knn_model.pkl','wb'))


############ Logistic Regression

# Create logistic regression model
lr = LogisticRegression()

# Define hyperparameters to tune
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search cross-validation
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_X, train_Y)

# Get best hyperparameters
best_params = grid_search.best_params_

# Use the best hyperparameters to train the model
best_lr = LogisticRegression(C=best_params['C'])
best_lr.fit(train_X, train_Y)

# Predict on test set
lr_y_pred = best_lr.predict(test_X)
lr_y_train_pred = best_lr.predict(train_X)

# Calculate accuracy
lr_accuracy = accuracy_score(test_Y, lr_y_pred)
lr_trainaccuracy = accuracy_score(train_Y,lr_y_train_pred)
lr_accuracy
lr_trainaccuracy
# The logistic regression model achieved an accuracy of approximately 0.87% on the test data and approximately 0.8505% on the training data. This indicates that the model is performing decently well on both the training and test sets, though the accuracy is slightly higher on the test set. 

# Saving Logistic Regression Model
pickle.dump(best_lr, open('logistic_regression_model.pk' ,'wb'))



###############  SVM

# SVC with linear kernel trick
model_linear = SVC(kernel = "linear")
model1 = model_linear.fit(train_X, train_Y)
pred_test_linear = model_linear.predict(test_X)
pred_train_linear = model_linear.predict(train_X)

# Calculate accuracy
accuracy_train = np.mean(pred_train_linear==train_Y)
accuracy_test = np.mean(pred_test_linear == test_Y)

accuracy_train
accuracy_test

# The accuracy on the training data is approximately 85%.
# The accuracy on the test data is approximately 87.2%.

### Hyperparameter Optimization
## RandomizedSearchCV

# Base model
model = SVC()

# Parameters set
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# Randomized Search Technique for exhaustive search for best model
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, 
                                  n_jobs = 3, cv = 3, scoring = 'accuracy', random_state = 0)
  
# Fitting the model for grid search
randomised = rand_search.fit(train_X, train_Y)

# Best parameters
randomised.best_params_
# The best parameters obtained were:-  Kernel: RBF  ,  Gamma: 1  and C: 10

# Best Model
best = randomised.best_estimator_

# Evaluate on Test data
pred_test = best.predict(test_X)

np.mean(pred_test == test_Y)
# Accuracy score of approximately 0.94% indicates that the model correctly predicted the target variable (downtime) for around 94% of the samples in the test set, showing an improvement compared to the previous accuracy score of 86.4%.

accuracy_svm = np.mean(pred_test == test_Y)
print("SVM  Accuracy:", accuracy_svm)
# The Support Vector Machine (SVM) model achieved a test accuracy of approximately 0.934%.

train_accuracy_svm = best.score(train_X, train_Y)
print("SVM Training Accuracy:", train_accuracy_svm)
# SVM model achieved a training accuracy of approximately 0.924%

# Saving Support Vector Classifier (SVC) Model
pickle.dump(best, open('svc_model.pkl','wb'))



############## GradientBoosting

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()
boost_clf1 = boost_clf.fit(train_X, train_Y)

grad_pred = boost_clf1.predict(test_X)

print(confusion_matrix(test_Y, grad_pred))
print(accuracy_score(test_Y, grad_pred))
# True Positive (TP): This represents the number of positive samples that were correctly classified as positive by the model. In this case, there are 248 instances where the model correctly predicted that the samples belong to the positive class.
# True Negative (TN): This represents the number of negative samples that were correctly classified as negative by the model. In this case, there are 245 instances where the model correctly predicted that the samples belong to the negative class.
# False Positive (FP): Also known as Type I error, this represents the number of negative samples that were incorrectly classified as positive by the model. In this case, there are 5 instances where the model incorrectly predicted that the negative samples belong to the positive class.
# False Negative (FN): Also known as Type II error, this represents the number of positive samples that were incorrectly classified as negative by the model. In this case, there are 2 instances where the model incorrectly predicted that the positive samples belong to the negative class.
# The accuracy score is 0.986, indicating that the model correctly predicted 0.986% of the test data samples.

print(confusion_matrix(train_Y, boost_clf1.predict(train_X)))
print(accuracy_score(train_Y,boost_clf1.predict(train_X)))
# True Positives (TP): There are 1011 instances where the model correctly predicted positive samples as positive.
# True Negatives (TN): There are 988 instances where the model correctly predicted negative samples as negative.
# False Positives (FP): There are no instances where negative samples were incorrectly classified as positive.
# False Negatives (FN): There is no instance where a positive sample was incorrectly classified as negative.
# The accuracy on the training data is approximately 1.0%.


# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)

boost_clf_p = boost_clf2.fit(train_X, train_Y)

grad_pred_p = boost_clf_p.predict(test_X)


# Evaluation on Testing Data

print(confusion_matrix(test_Y, grad_pred_p))
print('\n')
print(accuracy_score(test_Y,grad_pred_p))
# The accuracy on the testing data is approximately 0.99%.
# True Positives (TP): There are 247 instances where the model correctly predicted positive samples as positive.
# True Negatives (TN): There are 241 instances where the model correctly predicted negative samples as negative.
# False Positives (FP): There are 6 instances where negative samples were incorrectly classified as positive.
# False Negatives (FN): There are 0 instances where positive samples were incorrectly classified as negative.

# Evaluation on Training Data

print(confusion_matrix(train_Y, boost_clf_p.predict(train_X)))
accuracy_score(train_Y, boost_clf_p.predict(train_X))
# The accuracy on the training data is approximately 0.989%
# True Positives (TP): There are 977 instances where the model correctly predicted positive samples as positive.
# True Negatives (TN): There are 999 instances where the model correctly predicted negative samples as negative.
# False Positives (FP): There are 13 instances where negative samples were incorrectly classified as positive.
# False Negatives (FN): There are 11 instances where positive samples were incorrectly classified as negative.

# Save the ML model
pickle.dump(boost_clf_p, open('gradiantboostparam.pkl', 'wb'))



############   Naive bayes
# Create the Naive Bayes classifier
nb = GaussianNB()
nb.fit(train_X, train_Y)
# calculating train accuracy
nb_y_train_pred = nb.predict(train_X)
nb_trainaccuracy= accuracy_score(train_Y,nb_y_train_pred)
nb_trainaccuracy

# Calculating test accuracy
nb_y_pred = nb.predict(test_X)
nb_accuracy = accuracy_score(test_Y, nb_y_pred)
nb_accuracy

# For the Naive Bayes (NB) model:
# The accuracy on the training data is approximately 85%.
# The accuracy on the test data is approximately 0.872%.

# Saving Model
pickle.dump(nb, open('naive_bayes_model.pk' ,'wb'))


# Comparing accuracies 

# Decision tree
print("Decision Tree train Accurocy:",dt_train_accuracy)
print("Decision Tree  test Accurocy:", dt_accuracy)

# KNN
print("KNN accuracy on train data :", knn_train_accuracy)
print("kNN Accuracy on test data  :", knn_accuracy)

# RANDOM FOREST 
print(f'Random forest Train Accuracy: {rf_Grid.score(train_X, train_Y):.3f}')
print(f'Random forest Test Accuracy: {rf_Grid.score(test_X, test_Y):.3f}')

# SVM
print("SVM Training Accuracy:", train_accuracy_svm)
print("SVM  Test Accuracy:", accuracy_svm)

# Gradient boosting
print("gradient boosting Train Accuracy:",accuracy_score(train_Y, boost_clf_p.predict(train_X)))
print("gradient boosting Test Accuracy:",(accuracy_score(test_Y,grad_pred_p)))

# Naive bayes
print("Naive bayes train accuracy:",nb_trainaccuracy)
print("Naive bayes test accuracy:",nb_accuracy)

# Logistic regression
print("logistic regression train accuracy :",lr_trainaccuracy)
print("logistic regression test accuracy :",lr_accuracy)

# here ,we can notice that the Random Forest model typically exhibits high accuracy on both training and test datasets. In this case, the model achieved a training accuracy of 96.3% and a test accuracy of 96.6%, indicating robust performance in predicting downtime.
##############################################################################3
