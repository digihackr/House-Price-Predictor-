from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Cleaned_Data.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    location = sorted(data['location'].unique())
    return render_template('index.html', location = location)

@app.route('/predict' , methods = ['POST'])
def predict():
    location = request.form.get('location')
    BHK = request.form.get('BHK')
    bath = request.form.get('bath')
    total_sqft = request.form.get('total_sqft')

    print(location, BHK, bath, total_sqft)

    input = pd.DataFrame([[location,total_sqft,bath,BHK]], columns = ['location','total_sqft','bath','BHK'])
    prediction = pipe.predict(input)[0] * 100000

    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug = True, port = 5001)