from fastapi import FastAPI, File, UploadFile,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model
from datetime import datetime

import pandas as pd
import csv
import codecs
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

app = FastAPI()
origins = [
    "*",
    "https://localhost:3000/",
    "http://localhost:3000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_product_names(file: UploadFile):
    decoded_file = codecs.iterdecode(file.file, 'utf-8')
    csvReader = csv.reader(decoded_file)
    next(csvReader, None)  
    product_names = [row[0] for row in csvReader]  # Get the first column of each row
    return product_names

@app.get("/")
def read_root():
    return {"Hello": "Test Route"}

@app.post("/data")
async def read_data(background_tasks: BackgroundTasks,file:UploadFile=File(...)):
    product_names = get_product_names(file)
    look_back = 10
    sales = {}
    products = {}
    cwd = os.getcwd()
    print(cwd)
    with open(f'{cwd}/dataset/dataset.csv', mode ='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            sales[lines[0]] = [i for i in lines[1:53]]
        del(sales['Product_Code'])

    for product in product_names:
        try:
            dataset = {'dataset': [float(i) for i in sales[product]]+[float(i) for i in sales[product]]}
            kj = pd.date_range(end=datetime.today(), periods=len(dataset['dataset']), freq='W').tolist()
            dataset['kj'] = kj	
            df = pd.DataFrame(dataset['dataset'],index=dataset['kj'],columns =['Open'])
            model_path = (f"{cwd}/models/{product}.h5")
            model=load_model(model_path)
            data = df.values
            data = data.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(data)
            data_list = [item[0] for item in data]
            data_value = data.copy()

            for i in range(5):
                X_new = data_value[-look_back:]
                X_new = np.reshape(X_new, (1, 1, look_back))
                Y_new = model.predict(X_new,verbose=0)
                Y_new = scaler.inverse_transform(Y_new)
                data_value =  np.append(data_value,[float(Y_new),])

            data1 = scaler.inverse_transform(data)
            lis = [None for _ in range(len(data)-1)] + [int(data1[-1]),] + [int(i)-1 for i in data_value[-5:]]
            data_list = data_list.tolist() if isinstance(data_list, np.ndarray) else data_list
            lis = lis.tolist() if isinstance(lis, np.ndarray) else lis
            lis = ['null' if v is None else v for v in lis]

            products[product] = {
                'OldValue': data_list, 
                'NewValue': lis  
            }
        except:
            print("one product miss ")

    json_products = json.dumps(products,default=float)
    return {"data":json_products}