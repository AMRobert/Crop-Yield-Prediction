import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    Area = int(request.form['Area'])
    Area = np.log(Area)
    print(Area)
    int_features = [int(x) for x in request.form.values()]
    int_features.pop()
    int_features.insert(0,Area)
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    prediction = np.exp(prediction)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Yield is (in tons){}'.format(output))