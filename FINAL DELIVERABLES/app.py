from flask import render_template,Flask,request
import numpy as np
import pickle
import requests

API_KEY = "tC17RU_LNs8nvwyr_5MCc2Q8SgplCu8tINl45gMRtqJL"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app= Flask(__name__, template_folder='templates')

scale = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
     return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    hemo,rc,sg,al,sc,htn,sod,bp,wc,age = [float(x) for x in request.form.values()]

    features = [[hemo,rc,sg,al,sc,htn,sod,bp,wc,age]]
    #con_features = [np.array(features)]
    scale_features = scale.fit_transform(features)
    sf = scale_features.tolist()
    
    payload_scoring = {"input_data": [{"fields": ['hemo','rc','sg','al','sc','htn','sod','bp','wc','age'], "values": sf}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/212e54bb-1225-4f5d-a605-ab349d76bd3f/predictions?version=2022-11-14', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
    print("response_scoring")
    prediction = response_scoring.json()
    predict = prediction['prediction'][0]

    if prediction:
        return render_template('index.html',prediction_text='Oops! You have Chronic Kidney Disease.')
    else:
        return render_template('index.html',prediction_text="Great! You don't have Chronic Kidney Disease.")

if __name__ == "__main__":
    app.run(debug=True)