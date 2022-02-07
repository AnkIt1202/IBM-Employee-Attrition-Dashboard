from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

#Loading Trained Model
model = load_model('Final_model')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    #predict_model function takes a trained model object and the dataset to predict
    
    predictions = predictions_df['Label'][0]
    return predictions


st.title('Employee Attrition Prediction Web App')

satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
salary = st.selectbox('salary', ['low', 'high','medium'])
average_montly_hours = st.number_input('average_montly_hours',  min_value=96, max_value=310, value=100)
department = st.selectbox('department', ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'])


output=""

input_dict={'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'salary' : salary, 'average_montly_hours':average_montly_hours, 'department':department}


input_df = pd.DataFrame([input_dict])
print(input_df)

if st.button("Predict"):
	output = predict(model=model, input_df=input_df)
	output = str(output)
        
st.success('The output is {}'.format(output))

    