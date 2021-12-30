import os.path

import pandas as pd
import numpy as np
from flask import Flask,render_template,request,jsonify,url_for
import pickle




# picFolder=os.path.join('static','images')
# app.config

app = Flask(__name__)
model1= pickle.load(open('model.pkl','rb'))
diabetesmodel = pickle.load(open('diabetesmodel.pkl','rb'))
physicalDiabetes = pickle.load(open('physicalDiabetes.pkl','rb'))
diabetesmodelscaler = pickle.load(open('diabetesmodelscaler.pkl','rb'))

# path="../DATASETS/Disease/diabaties/scalerdiabetes.csv"
# di=pd.read_csv(path)



@app.route("/")
def Home():
    return render_template('index.html')


# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     int_features=[float(x) for x in request.form.values()]
#     print(int_features)
#     final_features=[np.array(int_features)]
#     print(final_features)
#     prediction = model1.predict(final_features)
#     def nts(arg):
#         switcher = {1: 'Iris-versicolor', 2: 'Iris-setosa', 3: 'Iris-virginica'}
#         return switcher[arg]
#     return render_template('iris.html',prediction_text="\nThe predicted flower name with {:.2f} % accuracy is '{}'".format(9.96,nts(prediction[0])))


@app.route('/diabetespredict',methods=['POST','GET'])
def diabetespredict():
    input_data=[float(x) for x in request.form.values()]
    inumpyarray = np.asarray(input_data)
    ireshape = inumpyarray.reshape(1, -1)
    final_features = diabetesmodelscaler.transform(ireshape)
    prediction = diabetesmodel.predict(final_features)
    pre=""
    if(prediction[0]==1):
        pre="diabetes"
    else:
        pre="No diabetes"
    return render_template('diabetes.html',prediction_textd="\nFrom The report we know person have '{}'".format(pre))
# physicaldiabetes

@app.route('/physicaldiabetespredict',methods=['POST','GET'])
def physicaldiabetespredict():
    input_data=[float(x) for x in request.form.values()]
    print(input_data)
    inumpyarray = np.asarray(input_data)
    ireshape = inumpyarray.reshape(1, -1)
    final_features = ireshape
    prediction = physicalDiabetes.predict(final_features)
    pre=""
    if(prediction[0]==1):
        pre="diabetes"
    else:
        pre="No diabetes"
    return render_template('physical.html',physical_text="\nFrom The report we know person have '{}'".format(pre))

# @app.route('/iris',methods=['GET','POST'])
# def iris():
#     return render_template('iris.html')

@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route('/physicaldiabetes',methods=['GET','POST'])
def physicaldiabetes():
    return render_template('physical.html')



# main driver function
if __name__ == '__main__':
    app.run(debug=True)
