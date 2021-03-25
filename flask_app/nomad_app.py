from flask import Flask, Response, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mpld3
import pickle
import seaborn as sns

# initialize the flask app
app = Flask("myApp")

# route 1: forecast dashboard
@app.route("/")
def send_instructors():
    data0 = instructors(0)
    data1 = instructors(1)
    data2 = instructors(2)
    data3 = instructors(3)

    data = {'instructors': [{'Dave': data0}, {"Alice": data1}, {"Hamilton": data2}, {"Stevia": data3}]}
    return data

def instructors(inst_num):
    forecasting = pd.read_json('./assets/data/jewelry_json.json')
    forecasting['event_time'] = pd.to_datetime(forecasting['event_time'])
    forecasting.set_index("event_time", inplace=True)
    forecasting.sort_index(inplace=True)
    forecasting.price.replace(0, 1, inplace=True)
    forecasting = forecasting[forecasting.index.year.isin([2020])]
    prior_data = pd.DataFrame(forecasting)

    model = pickle.load(open("./assets/model/hw_pickle.pkl", "rb"))
    
    instructor = forecasting[forecasting['instructor_id'] == inst_num]
    forecaster = pd.DataFrame(instructor['price'].resample('W').sum())
    forecaster = forecaster.reset_index()
    forecaster.price.replace(0, 1, inplace=True)

    hw_predictions = model.forecast(6) # 6 week outlook
   
    forecaster = dict(forecaster['price'])
    forecaster = list(forecaster.values())

    forecaster.append(hw_predictions[0])
    forecaster.append(hw_predictions[1])
    forecaster.append(hw_predictions[2])
    forecaster.append(hw_predictions[3])
    forecaster.append(hw_predictions[4])
    forecaster.append(hw_predictions[5])

    return forecaster

if __name__ == "__main__":
    app.run(debug=True)