from application import app
from flask import Flask, send_file, render_template, request
import io 
import base64 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd 
import json 
import plotly 
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

import pickle

df = pd.read_csv('./weatherUAS_bersih.csv')

# Tentang Dataset
@app.route("/home")
@app.route("/")
def home():
    # graph1
    fig1 = px.histogram(df, x='Year', y='Rainfall',
     color='RainTomorrow', title='Rain Tomorrow (2008 - 2016)')

    graph1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # graph2
    fig2 = px.histogram(df, x='Month', y='Rainfall',
     color='RainTomorrow', title='Rain Tomorrow (January - December)')

    graph2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # graph3
    fig3 = px.histogram(df, x='Day', y='Rainfall',
     color='RainTomorrow', title='Rain Tomorrow')

    graph3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('home.html', graph1=graph1, graph2=graph2, graph3=graph3)

# Data Distribution
@app.route("/data")
def data():
    feature = 'RainToday'
    RainToday = create_plot(feature)
    return render_template('data.html', plot=RainToday)

def create_plot(feature):
    if feature == 'RainToday':
        x = df['RainToday']
        hist_data = [x]
        group_labels = ['RainToday']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='RainToday Distribution')

    elif feature == 'RainTomorrow':
        x = df['RainTomorrow']
        hist_data = [x]
        group_labels = ['RainTomorrow']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='RainTomorrow Distribution')

    elif feature == 'Date':
        x = df['Date']
        hist_data = [x]
        group_labels = ['Date']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='Date Distribution')

    elif feature == 'Location':
        x = df['Location']
        hist_data = [x]
        group_labels = ['Location']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='Location Distribution')

    elif feature == 'MinTemp':
        x = df['MinTemp']
        hist_data = [x]
        group_labels = ['MinTemp']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='MinTemp Distribution')

    elif feature == 'MaxTemp':
        x= df['MaxTemp']
        hist_data = [x]
        group_labels = ['MaxTemp']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='MaxTemp Distribution')

    elif feature == 'Rainfall':
        x= df['Rainfall']
        hist_data = [x]
        group_labels = ['Rainfall']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Rainfall Distribution')

    elif feature == 'Evaporation':
        x= df['Evaporation']
        hist_data = [x]
        group_labels = ['Evaporation']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Evaporation Distribution')
    
    elif feature == 'Sunshine':
        x= df['Sunshine']
        hist_data = [x]
        group_labels = ['Sunshine']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Sunshine Distribution')

    elif feature == 'WindGustDir':
        x = df['WindGustDir']
        hist_data = [x]
        group_labels = ['WindGustDir']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='WindGustDir Distribution')

    elif feature == 'WindGustSpeed':
        x= df['WindGustSpeed']
        hist_data = [x]
        group_labels = ['WindGustSpeed']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='WindGustSpeed Distribution')

    elif feature == 'WindDir9am':
        x = df['WindDir9am']
        hist_data = [x]
        group_labels = ['WindDir9am']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='WindDir9am Distribution')

    elif feature == 'WindDir3pm':
        x = df['WindDir3pm']
        hist_data = [x]
        group_labels = ['WindDir3pm']
        data = px.histogram(x=hist_data)
        data.update_layout(title_text='WindDir3pm Distribution')

    elif feature == 'WindSpeed9am':
        x= df['WindSpeed9am']
        hist_data = [x]
        group_labels = ['WindSpeed9am']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='WindSpeed9am Distribution')

    elif feature == 'WindSpeed3pm':
        x= df['WindSpeed3pm']
        hist_data = [x]
        group_labels = ['WindSpeed3pm']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='WindSpeed3pm Distribution')

    elif feature == 'Humidity9am':
        x= df['Humidity9am']
        hist_data = [x]
        group_labels = ['Humidity9am']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Humidity9am Distribution')

    elif feature == 'Humidity3pm':
        x= df['Humidity3pm']
        hist_data = [x]
        group_labels = ['Humidity3pm']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Humidity3pm Distribution')

    elif feature == 'Pressure9am':
        x= df['Pressure9am']
        hist_data = [x]
        group_labels = ['Pressure9am']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Pressure9am Distribution')

    elif feature == 'Pressure3pm':
        x= df['Pressure3pm']
        hist_data = [x]
        group_labels = ['Pressure3pm']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Pressure3pm Distribution')

    elif feature == 'Cloud9am':
        x= df['Cloud9am']
        hist_data = [x]
        group_labels = ['Cloud9am']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Cloud9am Distribution')

    elif feature == 'Cloud3pm':
        x= df['Cloud3pm']
        hist_data = [x]
        group_labels = ['Cloud3pm']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Cloud3pm Distribution')

    elif feature == 'Temp9am':
        x= df['Temp9am']
        hist_data = [x]
        group_labels = ['Temp9am']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Temp9am Distribution')

    else:
        x= df['Temp3pm']
        hist_data = [x]
        group_labels = ['Temp3pm']
        data = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
        data.update_layout(title_text='Temp3pm Distribution')

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/plotting', methods=['GET', 'POST'])
def change_features():
    feature = request.args['selected']
    graphJSON= create_plot(feature)

    return graphJSON

# Modelling
final_df = pd.read_csv('./datax.csv')

scaler = StandardScaler()
datax = final_df.drop(['RainTomorrow'],axis=1)

# Scaler data
X = scaler.fit_transform(datax)
y = final_df['RainTomorrow']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelling Logistic
logres = LogisticRegression()
logres.fit(x_train, y_train)
LogisticRegression(C=1.0,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                l1_ratio=None,
                max_iter=100,
                multi_class='auto',
                n_jobs=None,
                penalty='l2',
                random_state=None,
                solver='lbfgs',
                tol=0.0001,
                verbose=0,
                warm_start=False)

lr_pred = logres.predict(x_test)
score = accuracy_score(y_test, lr_pred)
MAPE = mean_absolute_percentage_error(y_test, lr_pred)
MAE = mean_absolute_error(y_test, lr_pred)
RMSE = sqrt(mean_squared_error(y_test, lr_pred))

# Logistic
y_pred_logistic = logres.predict_proba(x_test)[:,1]
logistic_fpr, logistic_tpr, threshold = metrics.roc_curve(y_test, y_pred_logistic)
logistic_auc = metrics.auc(logistic_fpr, logistic_tpr)

roc_df = pd.DataFrame(zip(logistic_fpr, logistic_tpr, threshold),columns = ["FPR","TPR","Threshold"])

@app.route('/callback2', methods=['POST', 'GET'])
def cb2():
    return gm(request.args.get('data'))

@app.route("/model")
def model():
    #create ROC curve
    plot_pred = px.area(roc_df, x='FPR', y='TPR')
    plot_pred.update_layout(title_text='Logistic Regression')

    plot_pred.add_annotation(x=0.5, y=0.5,
                             text=f"AUC={logistic_auc:.2f}", showarrow=False)

    graphPRED = json.dumps(plot_pred, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('model.html', score=score, MAPE=MAPE, MAE=MAE, RMSE=RMSE, graphPRED=graphPRED)

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        RainToday = request.form['RainToday']
        Cloud9am = request.form['Cloud9am']
        Cloud3pm = request.form['Cloud3pm']
        Humidity9am = request.form['Humidity9am']
        Humidity3pm = request.form['Humidity3pm']
        Rainfall = request.form['Rainfall']

        predict_list = [RainToday, Cloud9am, Cloud3pm, Humidity9am, Humidity3pm, Rainfall]
        # prediction = logres.predict(predict)
        sample = np.array(predict_list).reshape(1,-1)
        prediction = logres.predict(sample)
        
        output = {0: 'tidak akan hujan.', 1: 'akan hujan!'}

        return render_template('predict.html', prediction_text='Besok {}'.format(output[prediction[0]]))
    
    else:
        return render_template('predict.html')