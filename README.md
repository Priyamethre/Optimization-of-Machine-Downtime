
#  Optimization of Machine Downtime

## **Overview**  
Unplanned machine downtime is a significant challenge that leads to lost productivity, delayed deliveries, and increased maintenance costs. This project aims to address these issues by developing a predictive model to forecast machine failures before they occur. By taking proactive measures, companies can minimize downtime, reduce financial losses, and improve overall operational efficiency.

## **Business Problem**  
Machine downtime, especially in industries such as vehicle fuel pump manufacturing, results in:  
- **Significant financial losses** due to halted production.  
- **Reduced productivity** and inability to meet delivery deadlines.  
- **Increased maintenance costs** from unexpected breakdowns.  

**Objective:** Develop a predictive model that accurately forecasts machine failures, enabling timely maintenance and avoiding costly downtime.

## ** Approach**  
1. **Data Collection and Understanding:**  
   - **Collaboration:** Worked closely with the client to identify key machine parameters impacting downtime.  
   - **Key Parameters:**  
     - Spindle Bearing Temperature  
     - Spindle Vibration  
     - Tool Vibration  
     - Spindle Speed  
     - Voltage  
     - Torque  
   - **Dataset Details:**  
     - Provided dataset contains **20,000 rows** of machine operation data.
   - **Data Collection Process:**  
     - Data was sourced directly from the machine operation logs.
     - Collaborated with the client’s engineering team to ensure that all critical parameters were captured.
     - Validated the dataset through initial exploratory checks and ensured it covered various operating conditions.

2. **Exploratory Data Analysis (EDA):**  
   - Conducted a thorough EDA using visualizations (bar plots, scatter plots, and heatmaps) to understand the distribution of variables and their relationships with machine downtime.  
   - **Key Insights:**  
     - Higher Spindle Bearing Temperature and increased Spindle Speed were correlated with higher downtime rates.

3. **Data Cleaning and Preprocessing:**  
   - **Handling Missing Values:** Used imputation techniques (mean, median, and MICE) to fill missing data.  
   - **Outlier Treatment:** Applied Winsorization and capping to control the influence of outliers.  
   - **Feature Engineering:** Transformed date-time columns into actionable metrics, such as operation duration or time since the last maintenance session.

4. **Data Scaling:**  
   - Standardized all features using Min-Max Scaling and Standard Scaling to ensure balanced contribution across variables.

5. **Model Training and Evaluation:**  
   - Split the dataset into training and test sets.  
   - Experimented with multiple machine learning algorithms including Decision Tree, K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), Gradient Boosting, Naive Bayes, and Logistic Regression.  
   - Conducted over **120 experiments** using Grid Search for hyperparameter tuning.

6. **Model Selection:**  
   - **Best Model:** Random Forest  
     - **Training Accuracy:** 96.3%  
     - **Test Accuracy:** 96.6%  
   - This model demonstrated robust performance in predicting machine downtime, meeting the project's objectives.

7. **Deployment and Client Handoff:**  
   - Deployed the final model into a production environment.
   - Provided detailed documentation and guidelines for the client to test and validate the model on new data.

8. **Review of Success Criteria:**  
   - Achieved high prediction accuracy.
   - Demonstrated significant reduction in unexpected downtimes.
   - Delivered measurable cost savings and improved operational efficiency.

## **Technologies Used**  
- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-Learn (Random Forest, Decision Tree, KNN, SVM, Gradient Boosting, Naive Bayes, Logistic Regression)  
- **Preprocessing:** Imputation (Mean, Median, MICE), Winsorization, Scaling (Min-Max, Standard)

## **Conclusion**  
This project successfully addressed the problem of unplanned machine downtime by developing a predictive model that accurately forecasts machine failures. The Random Forest model, with a training accuracy of 96.3% and test accuracy of 96.6%, demonstrates that proactive measures can be taken to minimize downtime, improve productivity, and reduce costs. The balanced nature of the dataset further ensured that the model could be trained effectively without additional resampling, providing reliable and actionable insights for maintenance scheduling and operational efficiency.
