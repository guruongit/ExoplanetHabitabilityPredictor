from flask import Flask, render_template,request
from joblib import dump, load
import numpy as np
import sklearn
import pickle



model = pickle.load(open('modelKNN.pkl','rb'))
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_Ex():
    P_PERIOD = float(request.form.get('P_PERIOD'))
    P_FLUX = float(request.form.get('P_FLUX'))
    P_TEMP_EQUIL = float(request.form.get('P_TEMP_EQUIL'))
    P_TYPE   = float(request.form.get('P_TYPE'))
    P_HABZONE_OPT  = float(request.form.get('P_HABZONE_OPT'))
    P_RADIUS_EST   = float(request.form.get('P_RADIUS_EST'))
    P_MASS_EST   = float(request.form.get('P_MASS_EST'))
    S_TYPE_TEMP = float(request.form.get('S_TYPE_TEMP'))

    result = model.predict(np.array([P_PERIOD , P_FLUX ,P_TEMP_EQUIL ,P_TYPE  ,P_HABZONE_OPT  ,P_RADIUS_EST ,   P_MASS_EST ,  S_TYPE_TEMP, ]).reshape(1,8))
    if result[0] ==1:
        result = 'Habitable'
    else:
        result = 'not place'
    return render_template('index.html',result= result)
    # return str(result)

if __name__ == '__main__':
    app.run(debug=True)