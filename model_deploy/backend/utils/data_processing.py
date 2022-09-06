import numpy as np
import pandas as pd
from csv import writer
from datetime import datetime

#To do: In the future create a airflow DAG for it
def save_log_prediction(input_data, out_data):
    #Get current date time
    now = datetime.now()
    data = np.column_stack((input_data, out_data))
    df = pd.DataFrame(i for i in data)
    df.columns = ["Pregnancies" ,"Glucose" ,"BloodPressure" ,"SkinThickness" ,"Insulin" ,"BMI" ,"DiabetesPedigreeFunction" ,"Age", "Predict"]
    df['Datetime'] = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('/app/backend/data/output/logs_prediction.csv', 'a', newline='\n', encoding='utf-8') as f_object:  
        _object = writer(f_object, delimiter =';')
        _object.writerows(df.values)  
        f_object.close()