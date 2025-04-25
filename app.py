import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import geopy
import os



from geopy.distance import geodesic


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Step 1: Create or load your dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Save the model
joblib.dump(model, "fraud_detection_model.jb")
print("Model trained and saved successfully.")





csv_path = r"C:/Users/kamun/OneDrive\Desktop/project 2/fraudTest.csv"

if os.path.exists(csv_path) and os.access(csv_path, os.R_OK):
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully.")
else:
    print("File is missing or not readable.")



def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1),(lat2,lon2)).km



st.title("Fraud Detection System")
st.write("Enter the Transaction details Below")
merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude",format="%.6f")
long = st.number_input("Longitude",format="%.6f")
merch_lat = st.number_input("Merchant Latitude",format="%.6f")
merch_long = st.number_input("Merchant Longitude",format="%.6f")
hour = st.slider("Transaction Hour",0,23,12)
day =st.slider("Transaction Day",1,31,15)
month = st.slider("Transaction MOnth",1,12,6)
gender = st.selectbox("Gender",["Male","Female"])
cc_num = st.text_input("Credit Card number")

distance = haversine(lat,long,merch_lat,merch_long)

if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category,amt,distance,hour,day,month,gender, cc_num]],
                                  columns=['merchant','category','amt','distance','hour','day','month','gender','cc_num'])
        
        categorical_col = ['merchant','category','gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col]=-1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x:hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "Fraudulant Transaction" if prediction == 1 else " Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please Fill all required fields")