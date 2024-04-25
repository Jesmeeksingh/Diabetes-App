from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd
model = pickle.load(open('model.pkl','rb'))

diabtes_dataset  = pd.read_csv('diabetes.csv')
X = diabtes_dataset.drop(columns = 'Outcome',axis=1)
scaler = StandardScaler()
scaler.fit(X)

app = Flask(__name__)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/prediction',methods=['POST'])
def prediction():
    int_features = [float(x) for x in request.form.values()]

    features = np.array(int_features)
    features_reshaped = features.reshape(1,-1)
    for i in range(len(features_reshaped)):
        print(features_reshaped[i])
    standardized_data = scaler.transform(features_reshaped)
    ans = model.predict(standardized_data)
    print(ans)
    if ans ==1:
        return render_template('is_diabetic.html') 
    elif ans == 0:
        return render_template('not_diabetic.html')
app.run()