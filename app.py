from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np # linear algebra
import pandas as pd
from datetime import datetime
import json
from flask import jsonify
pd.options.mode.chained_assignment = None  # default='warn'
import joblib
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import load_model
import xgboost as xgb
from multiprocessing import Process
from flask_cors import CORS, cross_origin
import calendar
import os
from flask import Flask, flash, request, redirect, url_for

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
ALLOWED_EXTENSIONS = set(['csv',])
UPLOAD_FOLDER = '/Uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	


@app.route('/api/v1/uploadertest', methods = ['GET', 'POST'],endpoint='func10')
@cross_origin()
def upload_file():
   if request.method == 'POST':
     
      file =  request.files['file']

      if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM2020.csv')
            file.save(savepath)
        
      return 'file uploaded successfully'     



@app.route('/api/v1/uploaderAnalysis', methods = ['GET', 'POST'],endpoint='func20')
@cross_origin()
def upload_file():
   if request.method == 'POST':
     # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
      file =  request.files['file']
      dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
      if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            savepath1sor=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'new.csv')
            file.save(savepath1sor)
            df= pd.read_csv(savepath1sor, date_parser=dateparse,index_col=0)
            if 'Cat' in df.columns:
              savepathdes=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'DatSetWithFeatureEncoding_new2.csv')
              os.rename(savepath1sor, savepathdes)
            
            else :
              savepathdes=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv')
              os.rename(savepath1sor, savepathdes)
      return 'file uploaded successfully'  




@app.route('/api/v1/GetData',  methods = ['GET', 'POST'],endpoint='func1')
@cross_origin()
def upload_file():
   dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
   savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM2020.csv')
   if savepath is not None:
    df= pd.read_csv(savepath, date_parser=dateparse,index_col=0)
    dataset=df[['Units','Year','Month','Day','MeanTemp','Day_Of_the_Week','Week_of_the_Month','Week_of_the_Year','IsWeekDay','Winter_State','Holidays','Event_SchoolHoliday']]
    newdf=addnewFeatures(dataset)
    newdf1=FeatureEncoding(newdf)
    ouptput=DoPrediction(newdf1)
   
  # ouptput=ouptput.reset_index()
   jsonfiles = json.loads(ouptput.to_json(orient='records'))
 #  return 

   # freqs = {
   #    'status': 200,
   #    'entities': ,
   #    'error':""
   # }

   return ouptput.to_json(orient='records')
  # return render_template('index.html', ctrsuccess=jsonfiles)

  # lists = ouptput.tolist()
  # json_str = json.dumps(lists)
  # df= pd.read_csv('LSTM_TEST_Daliy_New_v3.csv',parse_dates=["Date"], date_parser=dateparse)
 # dataset=df[['Units','Year','Month','Day','MeanTemp','Day_Of_the_Week','Week_of_the_Month','Week_of_the_Year','IsWeekDay','Winter_State','Holidays','Event_SchoolHoliday']]
 #  return json_str
   #return ouptput.to_json(orient="records")
  # return ouptput.to_json(orient="records")
 
   
def addnewFeatures(dataset):
   dataset['is_month_start']=dataset.index.is_month_start.astype(int)
   dataset['is_month_end']=dataset.index.is_month_end.astype(int)
   dataset['quarter']=dataset.index.quarter
   dataset['is_quarter_start']=dataset.index.is_quarter_start.astype(int)
   dataset['is_quarter_end']=dataset.index.is_quarter_end.astype(int)
   dataset['is_year_end']=dataset.index.is_year_end.astype(int)
   dataset['is_year_start']=dataset.index.is_year_start.astype(int)
   dataset['is_leap_year']=dataset.index.is_leap_year.astype(int)
   dataset['Week_of_the_Year']=dataset.index.isocalendar().week.astype(int)
   dataset['Day_Of_the_Week']=dataset.index.dayofweek.astype(int)
   dataset['rolling_mean'] = dataset['Units'].rolling(window=7).mean()
   dataset['expanding_mean'] = dataset['Units'].expanding(7).mean()
   dataset['expanding_mean']=dataset['expanding_mean'].fillna(0)
   dataset['rolling_mean']=dataset['rolling_mean'].fillna(0)

   return dataset

def FeatureEncoding(dataset):
    
   scaler1=joblib.load('lstm_model.plk')
   scaled_features = dataset.copy()
   scaler1 = MinMaxScaler()
   col_names1  = ['MeanTemp']
   trasfeatures1 = scaled_features[col_names1]
   scaler1.fit(trasfeatures1.values)
   trasfeatures1 = scaler1.transform(trasfeatures1.values)
   dataset['MeanTemp']=trasfeatures1

   dataset['UnitsLag_0'] = dataset['Units'].shift(1)
   dataset['UnitsLag_0']=dataset['UnitsLag_0'].fillna(0)
   dataset['UnitsLag_2'] = dataset['Units'].shift(6)
   dataset['UnitsLag_2']=dataset['UnitsLag_2'].fillna(0)
   dataset['UnitsLag_5'] = dataset['Units'].shift(13)
   dataset['UnitsLag_5']=dataset['UnitsLag_5'].fillna(0)
   dataset['UnitsLag_8'] = dataset['Units'].shift(20)
   dataset['UnitsLag_8']=dataset['UnitsLag_8'].fillna(0)
   dataset['UnitsLag_11'] = dataset['Units'].shift(27)
   dataset['UnitsLag_11']=dataset['UnitsLag_11'].fillna(0)
   dataset['UnitsLag_12'] = dataset['Units'].shift(28)
   dataset['UnitsLag_12']=dataset['UnitsLag_12'].fillna(0)

   return dataset

def DoPrediction(dataset):
   model = xgb.XGBRegressor()
   model.load_model('xgb.bin')
   X_test, y_test = create_features(dataset, label='Units')
   dataset['MW_Prediction'] = model.predict(X_test)
   dataset['Date'] =  dataset['Year'].map(str) +"-" + dataset['Month'].map(str) +"-" + dataset['Day'].map(str)
  # forecast_index1=pd.date_range(start='2020-01-03',periods=30)
  # forecast_df1=pd.DataFrame(data=forecast1,index=forecast_index1,columns=['Units'])

  # forecast_index1=pd.date_range(start='2020-01-03',periods=30)
   #forecast_df1=pd.DataFrame(data=forecast1,index=forecast_index1,columns=['Units'])
   return dataset[:14]
   
#def Preprocess(dataset):
  #n_future=1
  #n_past=2
  #n_features=28
  #train_x=[]
  #train_y=[]


  #n_traindays=round((len(dataset)))
  #values=dataset.values
  #train = values[:n_traindays, :]
  #for i in range(n_past,len(train)-n_future +1):
  #train_x.append(train[i-n_past:i,0:train.shape[1]])
  #train_y.append(train[i+n_future-1:i+n_future,0])
  #train_x1,train_y=np.array(train_x),np.array(train_y)
  #return train_x1


def DoPrediction2(dataset):
   model = xgb.XGBRegressor()
   model.load_model('xgb.bin')
   X_test, y_test = create_features(dataset, label='Units')
   dataset['MW_Prediction'] = model.predict(X_test)
   dataset['Date'] =  dataset['Year'].map(str) +"-" + dataset['Month'].map(str) +"-" + dataset['Day'].map(str)
  # forecast_index1=pd.date_range(start='2020-01-03',periods=30)
  # forecast_df1=pd.DataFrame(data=forecast1,index=forecast_index1,columns=['Units'])

  # forecast_index1=pd.date_range(start='2020-01-03',periods=30)
   #forecast_df1=pd.DataFrame(data=forecast1,index=forecast_index1,columns=['Units'])
   return dataset.tail(30)



def create_features(df, label=None):  
    X = df[['Year','Month','Day','MeanTemp','Day_Of_the_Week','Week_of_the_Month','Week_of_the_Year','IsWeekDay','Winter_State','Holidays','Event_SchoolHoliday','is_month_start'
    ,'is_month_end','quarter','is_quarter_start','is_quarter_end','is_year_end','is_year_start','is_leap_year','rolling_mean','expanding_mean','UnitsLag_0','UnitsLag_2','UnitsLag_5','UnitsLag_8','UnitsLag_11','UnitsLag_12']]
    if label:
        y = df[label]
        return X, y
    return X


@app.route('/api/v1/GetLearnData',  methods = ['GET', 'POST'],endpoint='func3')
@cross_origin()
def upload_file():
   dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
   savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv')
   if savepath is not None:
    df= pd.read_csv(savepath, date_parser=dateparse,index_col=0)
    dataset=df[['Units','Year','Month','Day','MeanTemp','Day_Of_the_Week','Week_of_the_Month','Week_of_the_Year','IsWeekDay','Winter_State','Holidays','Event_SchoolHoliday']]

    newdf=addnewFeatures(dataset)
    newdf1=FeatureEncoding(newdf)
    ouptput=DoPrediction2(newdf1)
    #jsonfiles = json.loads(ouptput.to_json(orient='records'))


   return ouptput.to_json(orient='records')


@app.route('/api/v1/Analysis',  methods = ['GET', 'POST'],endpoint='func4')
@cross_origin()
def upload_file():
   dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
   savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'DatSetWithFeatureEncoding_new2.csv')
   if savepath is not None:
    df= pd.read_csv(savepath, date_parser=dateparse,index_col=0)
   return df.to_json(orient='records')


@app.route('/api/v1/Analysis2',  methods = ['GET', 'POST'],endpoint='func5')
@cross_origin()
def upload_file():
   dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
   savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv')
   if savepath is not None:
    df= pd.read_csv(savepath, date_parser=dateparse,index_col=0)
    dataset=df[['Units','Year','Month','Day','MeanTemp','Day_Of_the_Week','Week_of_the_Month','Week_of_the_Year','IsWeekDay','Winter_State','Holidays','Event_SchoolHoliday']]
    dataset['Date'] =  dataset['Year'].map(str) +"-" + dataset['Month'].map(str) +"-" + dataset['Day'].map(str)
   # newdf=addnewFeatures(dataset)
   # newdf1=FeatureEncoding(newdf)
   # ouptput=DoPrediction2(newdf1)
   jsonfiles = json.loads(dataset.to_json(orient='records'))


   return dataset.to_json(orient='records')


@app.route('/api/v1/Analysis3',  methods = ['GET', 'POST'],endpoint='func6')
@cross_origin()
def upload_file():
   dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
   savepath=os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv')
   if savepath is not None:
    df= pd.read_csv(savepath, date_parser=dateparse,index_col=0)
    dataset=df[['Units','Year','Month']]
    dataset = dataset.groupby(['Year','Month']).agg({'Units': 'sum'}).reset_index()
    dataset['Month'] = dataset['Month'].apply(lambda x: calendar.month_abbr[x])
    dataset['Date'] =  dataset['Year'].map(str) +"-" + dataset['Month'].map(str) +"-01" 

   jsonfiles = json.loads(dataset.to_json(orient='records'))

   return dataset.to_json(orient='records')


@app.route('/api/v1/FileExsist',  methods = ['GET', 'POST'],endpoint='func19')
@cross_origin()
def checkFile():
   files = []
   savepath=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'DatSetWithFeatureEncoding_new2.csv'))
   savepathAnalyis1=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv'))
   savepathPred=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM2020.csv'))
  
 

   if savepath is True:
      files.append("Analysis File 2")
   
   if savepathPred is True:
       files.append("Prediction File") 

   if savepathAnalyis1 is True:
       files.append("Analysis File 1")  

 
   return json.dumps(files)


@app.route('/api/v1/Deletefile',  methods = ['GET', 'POST'],endpoint='func29')
@cross_origin()
def checkFile():
  
   values= request.args.get("Filename")

   savepath=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'DatSetWithFeatureEncoding_new2.csv'))
   savepathAnalyis1=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv'))
   savepathPred=os.path.exists(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM2020.csv'))
  
 

   if values is not None:
      if values=="AnalysisFile2" and savepath:
         os.remove(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'DatSetWithFeatureEncoding_new2.csv'))
      if values=="PredictionFile" and savepathPred:
         os.remove(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM2020.csv'))
      if values=="AnalysisFile1" and savepathAnalyis1:
         os.remove(os.path.join(app.root_path+app.config['UPLOAD_FOLDER'],'LSTM_TEST_Daliy_New_v3.csv'))


   return 'File Deleted Successfully'  
   # if savepathPred is True:
   #     files.append("Prediction File") 

   # if savepathAnalyis1 is True:
   #     files.append("Analysis File 1")  

		
if __name__ == '__main__':
   app.run()