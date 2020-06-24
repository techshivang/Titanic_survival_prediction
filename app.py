from flask import Flask,render_template,request,jsonify,make_response
import json
import pickle
import numpy as np

model=pickle.load(open('Titanic_Survival_Model.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    arr=np.array([[
        float(data['pclass']),
        float(data['age']),
        float(data['sibsp']),
        float(data['parch']),
        float(data['sex']),
        float(data['port'])
    ]])

    answer = model.predict(arr)[0]
    return {'survived': str(answer)}

if __name__=="__main__":
    app.run(debug=True)