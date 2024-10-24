
# pip install streamlit
import pandas as pd
import streamlit as st
import joblib
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the model
winsor = joblib.load('winsor')
scale = joblib.load('minmax')
imputation = joblib.load('meanimpute')

# Load the trained model
rn_model = joblib.load('rfc.pkl')

def preprocess_data(data):
    columns_to_transform = (['Hydraulic_Pressure(bar)',
           'Coolant_Pressure(bar)','Air_System_Pressure(bar)',
           'Spindle_Vibration(µm)',
           'Tool_Vibration(µm)','Spindle_Speed(RPM)','Voltage(volts)',
           'Torque(Nm)','Cutting(kN)'])  
    
# Apply the preprocessing pipeline to the specified columns
data_preprocessed = pd.DataFrame(imputation.transform(data),columns=columns_to_transform)
scaled_data = pd.DataFrame(scale.transform(data_preprocessed),columns=columns_to_transform)
#clean_data = pd.DataFrame(encoding.transform(scaled_data),columns=columns_to_transform)
scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
           'Air_System_Pressure(bar)', 
           'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
           'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']] = winsor.transform(scaled_data[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)',
                  'Air_System_Pressure(bar)', 
                  'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)',
                  'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)']])
   
    scaled_data = pd.DataFrame(winsor.transform(scaled_data),columns =columns_to_transform)
    scaled_data = scaled_data.drop(columns=['Air_System_Pressure(bar)', 
                                  'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Voltage(volts)'])
    
    return scaled_data


def predict_downtime(original ,data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    prediction = pd.DataFrame(rn_model.predict(data), columns=['predict downtime'])
    final = pd.concat([ prediction, original], axis=1)  # Concatenate with original data
    final.to_sql('downtime_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)
    return final

def main():
    st.markdown(
        "<h1 style='text-align: center; color:teal;'>Optimization of Machine Downtime</h1>",
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        "<h2 style='color: maroon;'>Machine Downtime Prediction</h2>",
        unsafe_allow_html=True
    )
    
    st.sidebar.subheader("Upload Data")
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:      
                data = pd.DataFrame()
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
   
    user = st.sidebar.text_input("user")
    pw = st.sidebar.text_input("password" , type = 'password')
    db = st.sidebar.text_input("database")
   
    result = ""
   
    if st.button("Run Prediction"):
        if data.empty:
            st.warning("Please upload a valid dataset.")
        else:
           preprocessed_data = preprocess_data(data)
           result = predict_downtime(data ,preprocessed_data, user, pw, db)
           import seaborn as sns
           cm = sns.light_palette("blue", as_cmap = True)
           st.table(result.style.background_gradient(cmap=cm))
                           
if __name__=='__main__':
    main()
