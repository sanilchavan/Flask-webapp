from flask import Flask,render_template,request
import pickle
import numpy as np 

filename = 'heart-diseases-model.pkl'

classifier = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain = int(request.form['chest pain type'])
        bp = int(request.form['resting blood pressure'])
        cholestrol = int(request.form['serum cholestoral in mg/dl'])
        sugar = int(request.form['fasting blood sugar'])
        cardio =  int(request.form['resting electrocardiographic results'])
        heart_rate = int(request.form['maximum heart rate'])
        angina = int(request.form['exercise induced angina'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope of the peak exercise'])
        major_vessels = int(request.form['number of major vessels'])
        thal = int(request.form['thal'])

        data = np.array([[age,sex,chest_pain,bp,cholestrol,sugar,cardio,heart_rate,angina,oldpeak,slope,major_vessels,thal]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html',prediction=my_prediction,)
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
    

