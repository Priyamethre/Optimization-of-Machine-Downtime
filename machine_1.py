
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

# Importing the data 
df = pd.read_csv(r"Machine Downtime.csv")

######################   DATA UNDERSTANDING    

duplicates = df.duplicated()
print(duplicates.sum())

null_values = df.isnull()
null_count = null_values.sum()
print(null_count)
# There are null values present 
print(df)

df.describe()

# dropping Dates column
df.drop(columns=['Dates'], inplace=True)

df.Downtime.value_counts()
df.Machine_ID.value_counts()
df.Assembly_Line_No.value_counts()

#EDA using Autoviz
sweet_report = sv.analyze(df)
sweet_report.show_html('sweet_report.html')

##################   DATA CLEANING 

predictors = df.loc[:, df.columns != "Downtime"]
type(predictors)

predictors.columns

target = df["Downtime"]
type(target)

target

df.duplicated().sum()

numeric_features = df.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('Downtime')

# Drop categorical features
categorical_features.drop(columns=['Machine_ID','Assembly_Line_No'],inplace=True)

##  Missing values Analysis       

df.isnull().sum()

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'median'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
imputation = preprocessor.fit(predictors)

## Save the pipeline
joblib.dump(imputation, 'meanimpute')

cleandata = pd.DataFrame(imputation.transform(predictors), columns = numeric_features)
cleandata
cleandata.isnull().sum()
# all missing values have been imputed successfully in your data

## Scaling with MinMaxScaler   
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

scaled_data.describe()
scaled_data.columns
scaled_data.info()
scaled_data.head

############## Outlier Analysis     

scaled_data.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#### Outlier analysis: Columns are continuous, hence outliers are treated

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
                                 'Air_System_Pressure(bar)', 
                                 'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
                                 'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)'])

outlier = winsor.fit(scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
       'Air_System_Pressure(bar)', 
       'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
       'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']])

# Save the winsorizer model '
joblib.dump(outlier, 'winsor')

scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
       'Air_System_Pressure(bar)', 
       'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
       'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']] = outlier.transform(scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
              'Air_System_Pressure(bar)', 
              'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
              'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']])

scaled_data
# Re-checking if all outliers are treated
scaled_data.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# All outliers have been treated, missing values have been imputed, categorical data has been encoded, and the dataset is now prepared for model building.

##     GRAPHICAL REPRESENTATION     
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

# Histogram 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculate the number of features and define the layout of subplots
num_features = len(df.columns)
num_cols = 3  # Number of columns for subplots
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
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
for i in range(num_features, len(axes)):
    axes[i].axis('off')
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

#Feature Selection using Random Forest 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(scaled_data, target)

feature_importances = pd.DataFrame(rf.feature_importances_,index=scaled_data.columns,columns=['importance'])
feature_important = feature_importances.sort_values(by='importance', ascending=False)

#Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances.index, feature_importances ['importance'], color='skyblue')
plt.xlabel('Importance') 
plt.ylabel('Features')
plt.title('Feature Importance') 
plt.show()
threshold = 0.01

# the important features are are 'Hydraulic_Pressure(bar)','Coolant_Pressure(bar)','Torque(Nm)', 'Cutting(kN)' and 'Spindle_Speed(RPM)'

# Separate features (X) and target variable (y)
# Identified the crucial features impacting downtime for the given X value using Randomforest analysis.
X = scaled_data.drop(columns=['Air_System_Pressure(bar)', 'Coolant_Temperature',
'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
'Spindle_Vibration(µm)', 'Tool_Vibration(µm)','Voltage(volts)'])
X.columns

Y = target
Y

# Check the shape of X and y
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

#  Y is the target variable ,converting it into 1D array
Y = np.ravel(Y)

# Data Partition into Train and Test
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, stratify = Y,random_state=42)

############# Random Forest Model
rf_Model = RandomForestClassifier()

# #### Hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
max_features = ['auto', 'sqrt','log2', None]
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]

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

accuracy_train = np.mean(pred_train_linear==train_Y)
accuracy_test = np.mean(pred_test_linear == test_Y)
accuracy_train
accuracy_test
# The accuracy on the training data is approximately 85%.
# The accuracy on the test data is approximately 87.2%.

### Hyperparameter Optimization
## RandomizedSearchCV

model = SVC()
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, 
                                  n_jobs = 3, cv = 3, scoring = 'accuracy', random_state = 0)
  
# Fitting the model for grid search
randomised = rand_search.fit(train_X, train_Y)
randomised.best_params_
best = randomised.best_estimator_
pred_test = best.predict(test_X)

np.mean(pred_test == test_Y)
# Accuracy score of approximately 0.94% indicates that the model correctly predicted the target variable (downtime) for around 94% of the samples in the test set.

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
print(confusion_matrix(train_Y, boost_clf1.predict(train_X)))
print(accuracy_score(train_Y,boost_clf1.predict(train_X)))
# The accuracy on the training data is approximately 1.0%.

boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf_p = boost_clf2.fit(train_X, train_Y)
grad_pred_p = boost_clf_p.predict(test_X)

print(confusion_matrix(test_Y, grad_pred_p))
print('\n')
print(accuracy_score(test_Y,grad_pred_p))
# The accuracy on the testing data is approximately 0.99%.

print(confusion_matrix(train_Y, boost_clf_p.predict(train_X)))
accuracy_score(train_Y, boost_clf_p.predict(train_X))
# The accuracy on the training data is approximately 0.989%

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


nb_y_pred = nb.predict(test_X)
nb_accuracy = accuracy_score(test_Y, nb_y_pred)
nb_accuracy
# The accuracy on the training data is approximately 85%.
# The accuracy on the test data is approximately 0.872%.

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
