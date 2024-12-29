
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.PriceIntelligenceSystem.pipelines.prediction_pipeline import CustomData, PredictionPipeline
application = Flask(__name__)
app = application


@app.route('/', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
                    CRIM = float(request.form.get('CRIM')),
                    ZN = float(request.form.get('ZN') ),
                    INDUS = float(request.form.get('INDUS')),
                    CHAS = float(request.form.get('CHAS')),
                    NOX = float(request.form.get('NOX')),
                    RM = float(request.form.get('RM')),
                    Age = float(request.form.get('Age')),
                    DIS  = float(request.form.get('DIS')),
                    RAD  = float(request.form.get('RAD')),
                    TAX = float(request.form.get('TAX')),
                    PTRATIO = float(request.form.get('PTRATIO')),
                    B = float(request.form.get('B')),
                    LSTAT = float(request.form.get('LSTAT')),
                    )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictionPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('home.html', result = result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)
