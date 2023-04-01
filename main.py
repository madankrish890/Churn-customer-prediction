import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
# Load the trained model
model = joblib.load('svm_model1.pkl')

# Define the user interface
st.title('Customer Churn Prediction')
st.sidebar.header('User Input Features')

# Create the input feature form
def user_input_features():
    CreditScore = st.sidebar.slider('Credit Score', 300, 850, 500)
    Geography = st.sidebar.selectbox('Country', ['France', 'Germany', 'Spain'])
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Age = st.sidebar.slider('Age', 18, 100, 30)
    Tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance = st.sidebar.slider('Balance', 0, 250000, 50000)
    NumOfProducts = st.sidebar.slider('Number of Products', 1, 4, 2)
    HasCrCard = st.sidebar.selectbox('Has Credit Card?', ['Yes', 'No'])
    IsActiveMember = st.sidebar.selectbox('Is Active Member?', ['Yes', 'No'])
    EstimatedSalary = st.sidebar.slider('Estimated Salary', 0, 500000, 100000)
    data = {'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary}
    features = pd.DataFrame(data, index=[0])
    return features

# Get the user input
user_input = user_input_features()

# Preprocess the user input
user_input['HasCrCard'] = user_input['HasCrCard'].map({'Yes':1, 'No':0})
user_input['IsActiveMember'] = user_input['IsActiveMember'].map({'Yes':1, 'No':0})
user_input = pd.get_dummies(user_input, columns=['Geography', 'Gender'])
#lb=LabelEncoder()
#user_input['Geography']=lb.fit_transform(user_input['Geography'])
#user_input['Gender']=lb.fit_transform(user_input['Gender'])
# Predict the customer churn
prediction = model.predict(user_input)
#prediction_proba = model.predict_proba(user_input)

# Display the prediction result
st.subheader('Prediction Result')
if prediction == 0:
    st.write('The customer is likely to stay.')
else:
    st.write('The customer is likely to churn.')
#st.write('Prediction Probability:', prediction_proba)
