from flask import Flask, render_template, request
import pandas as pd
import joblib
#
# from joblib import dump
#
# dump(model, 'heart_disease_model.joblib')
# from joblib import load
#
# model = load('heart_disease_model.joblib')


import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load your trained model
model_rf = joblib.load('heart_disease.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(request.form[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    # Make a prediction using the trained model
    pred_user = model_rf.predict(user_DF)

    # Display the result on the result.html page
    return render_template('result.html', result=pred_user[0])

if __name__ == '__main__':
    app.run(debug=True)
