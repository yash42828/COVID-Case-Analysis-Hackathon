import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
death = pickle.load(open('death.pkl', 'rb'))
recovered = pickle.load(open('recovered.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    """int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    confirmed_df = pd.read_csv('time_series_19-covid-Confirmed.csv')
    deaths_df = pd.read_csv('time_series_19-covid-Deaths.csv')
    recoveries_df = pd.read_csv('time_series_19-covid-Recovered.csv')
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()"""
    
    a =  [str(x) for x in request.form.values()]
    if '/' in a[0]:
        res = a[0].split('/')
    elif '-' in a[0]:
        res = a[0].split('-')
    from datetime import date
    d0 = date(2020, 1, 22)
    d1 = date(int(res[2]), int(res[1]), int(res[0]))
    delta = d1 - d0
    #dates = 62
    #print(delta.days)
    future_forcast = np.array([i for i in range(delta.days)]).reshape(-1, 1)

    if request.method == 'POST':
            if request.form.get('total') == 'Predict':
                # pass
                prediction = model.predict(future_forcast)
                print(prediction[-1])
                result = int(prediction[-1])
                return render_template('index.html', prediction_text='Total No of Confirmed Cases will be: {}'.format(result))
            elif request.form.get('total1') == 'Predict1':
                prediction2 = death.predict(future_forcast)
                result2 = int(prediction2[-1])
                return render_template('index.html',prediction_text2='Total No of Death Cases will be: {}'.format(result2))
            elif request.form.get('total2') == 'Predict2':
                prediction3 = recovered.predict(future_forcast)
                result3 = int(prediction3[-1])
                return render_template('index.html',prediction_text3='Total No of Recovered Cases will be: {}'.format(result3))



if __name__ == "__main__":
    app.run(debug=True)