#Flask Libraries
import numpy
from flask import Flask, redirect, url_for, render_template, request
import requests
import urllib
from datetime import datetime
import datetime
#Numpy
import numpy as np
#for importing our keras model
import keras.models
import pandas as pd
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import *
#importing preprocessing libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#initalizing flask app
app = Flask(__name__)
app.secret_key = 'my secret and not your secret'


@app.route('/aboutus')
def about():
    return render_template("about.html")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/grantaccess', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        year = request.form['date'][:4]
        datetemp = datetime.datetime.strptime(request.form['date'], '%Y-%m-%d').strftime('%d/%m/%y')
        return redirect(url_for('jsonlocation', location=request.form['location'], date=datetemp, year=year))
    else:
        return redirect(url_for('index'))


@app.route('/location')
def jsonlocation():
    year = request.args.get('year')
    location = request.args.get('location')
    date = request.args.get('date')
    url = 'http://dev.virtualearth.net/REST/v1/Locations?'
    key = 'AozIVsiQ675xXwo2NwGtEuv8vtcQ098NSmpCuV1QAl7nFQ9wfjtcwSI_gdbH4sZV'
    cr = 'IN'
    results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
    response = requests.get(results)
    parser = response.json()
    print("jsonlocation response\n", parser)
    auth = parser['statusDescription']
    if auth == 'OK':
        if 'adminDistrict' not in parser['resourceSets'][0]['resources'][0]['address']:
            return 'Location does not exist in India! Please try again!'
        state = parser['resourceSets'][0]['resources'][0]['address']['adminDistrict']
        lat = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][0]
        lon = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][1]
        city = parser['resourceSets'][0]['resources'][0]['address']['locality']
        return redirect(url_for('jsonweather', city=city, state=state, date=date, latitude=lat, longitude=lon, year=year))
    else:
        return 'Status: %s! Server issue! Please try again later!' % auth
    

@app.route('/weather')
def jsonweather():
    year = request.args.get('year')
    city = request.args.get('city')
    state = request.args.get('state')
    date = request.args.get('date')
    lat = request.args.get('latitude')
    lon = request.args.get('longitude')
    url = 'https://api.openweathermap.org/data/2.5/forecast?'
    key = 'e31020243ddd05cc3d37ad5f4816190f'
    mode = 'json'
    count = 32
    results = url + urllib.parse.urlencode(({'lat': lat, 'lon': lon, 'appid': key, 'mode': mode, 'cnt': count}))
    response = requests.get(results)
    
    parser = response.json()
    print("jsonweather response\n", parser)
    auth = parser['cod']
    totalprecipitation = 0.00
    if auth == '200':
        for each in parser['list']:
            if 'rain' not in each:
                continue
            else:
                rain = each['rain']
                if '3h' in rain:
                    prec = rain['3h']
                    totalprecipitation += prec
    else:
        return 'Status: %s! Server issue! Please try again later!' % auth
    
    totalprecipitation = totalprecipitation/4
    return redirect(url_for('interim', city=city, state=state, date=date, precipitation=round(totalprecipitation, 2), year=year))


@app.route('/interim', methods=['GET', 'POST'])
def interim():
    
    
    city = request.args.get('city')
    state = request.args.get('state')
    state_symbol = state
    date = request.args.get('date')
    precipitation = float(request.args.get('precipitation'))
    #calculated
    month = datetime.datetime.strptime(date, '%d/%m/%y').strftime('%m')
    actualMonth = month
    year = request.args.get('year')
    terrain = ""
    duration = 0
    
    
    #for the month
    if(month == '01' or month == '02'):
        month = 'Jan-Feb'
        duration = 2
    elif(month == '03' or month == '04' or month == '05'):
        month = 'Mar-May'
        duration = 3
    elif(month == '06' or month == '07' or month == '08' or month == '09'):
        month = 'Jun-Sep'
        duration = 4
    elif(month == '10' or month == '11' or month == '12'):
        month = 'Oct-Dec'
        duration = 3
    
    #for the state code
    if(state == 'WB'):
        state = 'GANGETIC WEST BENGAL'
    elif(state == 'AN'):
        state = 'ANDAMAN & NICOBAR ISLANDS'
    elif(state == 'AR'):
        state = 'ARUNACHAL PRADESH'
    elif(state == 'AS'):
        state = 'ASSAM & MEGHALAYA'
    elif(state == 'BR'):
        state = 'BIHAR'
    elif(state == 'CG'):
        state = 'CHATTISGARH'
    elif(state == 'AD'):
        state = 'COASTAL ANDHRA PRADESH'
    elif(state == 'KA'):
        state = 'COASTAL KARNATAKA'
    elif(state == 'MP'):
        state = 'EAST MADHYA PRADESH'
    elif(state == 'RJ'):
        state = 'EAST RAJASTHAN'
    elif(state == 'UP'):
        state = 'EAST UTTAR PRADESH'
    elif(state == 'GJ'):
        state = 'GUJARAT REGION'
    elif(state == 'DL'):
        state = 'HARYANA DELHI & CHANDIGARH'
    elif(state == 'HP'):
        state = 'HIMACHAL PRADESH'
    elif(state == 'JK'):
        state = 'JAMMU & KASHMIR'
    elif(state == 'JH'):
        state = 'JHARKHAND'
    elif(state == 'KL'):
        state = 'KERALA'
    elif(state == 'GA'):
        state = 'KONKAN & GOA'
    elif(state == 'LD'):
        state = 'LAKSHWADEEP'
    elif(state == 'MH'):
        state = 'MADHYA MAHARASHTRA'
    elif(state == 'MT'):
        state = 'MATATHWADA'
    elif(state == 'MN'):
        state = 'NAGA MANI MIZO TRIPURA'
    elif(state == 'KA'):
        state = 'NORTH INTERIOR KARNATAKA'
    elif(state == 'OD'):
        state = 'ORISSA'
    elif(state == 'PB'):
        state = 'PUNJAB'
    elif(state == 'RS'):
        state = 'RAYALSEEMA'
    elif(state == 'SK'):
        state = 'SAURASHTRA & KUTCH'
    elif(state == 'KA'):
        state = 'SOUTH INTERIOR KARNATAKA'
    elif(state == 'SK'):
        state = 'SUB HIMALAYAN WEST BENGAL & SIKKIM'
    elif(state == 'TN'):
        state = 'TAMIL NADU'
    elif(state == 'TS'):
        state = 'TELANGANA'
    elif(state == 'UK'):
        state = 'UTTARAKHAND'
    elif(state == 'VD'):
        state = 'VIDARBHA'
    elif(state == 'MP'):
        state = 'WEST MADHYA PRADESH'
    elif(state == 'RJ'):
        state = 'WEST RAJASTHAN'
    elif(state == 'UP'):
        state = 'WEST UTTAR PRADESH' 
      
    #for the terrain
    if(state == 'GANGETIC WEST BENGAL'):
        terrain = 'Coastal-plateau'
    elif(state == 'ANDAMAN & NICOBAR ISLANDS'):
        terrain = 'Island'
    elif(state == 'ARUNACHAL PRADESH'):
        terrain = 'Hilly'
    elif(state == 'ASSAM & MEGHALAYA'):
        terrain = 'Hilly'
    elif(state == 'BIHAR'):
        terrain = 'Plain-land'
    elif(state == 'CHATTISGARH'):
        terrain = 'Hilly'
    elif(state == 'COASTAL ANDHRA PRADESH'):
        terrain = 'Coastal'
    elif(state == 'COASTAL KARNATAKA'):
        terrain = 'Coastal'
    elif(state == 'EAST MADHYA PRADESH'):
        terrain = 'Everything'
    elif(state == 'EAST RAJASTHAN'):
        terrain = 'Desert'
    elif(state == 'EAST UTTAR PRADESH'):
        terrain = 'Rugged'
    elif(state == 'GUJARAT REGION'):
        terrain = 'Desert/marsh'
    elif(state == 'HARYANA DELHI & CHANDIGARH'):
        terrain = 'Plain-land'
    elif(state == 'HIMACHAL PRADESH'):
        terrain = 'Hilly'
    elif(state == 'JAMMU & KASHMIR'):
        terrain = 'Hilly'
    elif(state == 'JHARKHAND'):
        terrain = 'Forest'
    elif(state == 'KERALA'):
        terrain = 'Coastal'
    elif(state == 'KONKAN & GOA'):
        terrain = 'Hilly/coastal'
    elif(state == 'LAKSHWADEEP'):
        terrain = 'Island'
    elif(state == 'MADHYA MAHARASHTRA'):
        terrain = 'Plain-land'
    elif(state == 'MATATHWADA'):
        terrain = 'Barren'
    elif(state == 'NAGA MANI MIZO TRIPURA'):
        terrain = 'Hilly'
    elif(state == 'NORTH INTERIOR KARNATAKA'):
        terrain = 'Coastal'
    elif(state == 'ORISSA'):
        terrain = 'Coastal'
    elif(state == 'PUNJAB'):
        terrain = 'Plain-land'
    elif(state == 'RAYALSEEMA'):
        terrain = 'Plain-land'
    elif(state == 'SAURASHTRA & KUTCH'):
        terrain = 'Hilly'
    elif(state == 'SOUTH INTERIOR KARNATAKA'):
        terrain = 'Coastal'
    elif(state == 'SUB HIMALAYAN WEST BENGAL & SIKKIM'):
        terrain = 'Hilly'
    elif(state == 'TAMIL NADU'):
        terrain = 'Hilly/coastal'
    elif(state == 'TELANGANA'):
        terrain = 'Hilly/plain'
    elif(state == 'UTTARAKHAND'):
        terrain = 'Hilly'
    elif(state == 'VIDARBHA'):
        terrain = 'Plain-land'
    elif(state == 'WEST MADHYA PRADESH'):
        terrain = 'Plain-land'
    elif(state == 'WEST RAJASTHAN'):
        terrain = 'Desert'
    elif(state == 'WEST UTTAR PRADESH'):
        terrain = 'Hilly'
        
    return redirect(url_for('predict', city=city, state=state, month=month, precipitation=precipitation, duration=duration, terrain=terrain, year=year, actualMonth=actualMonth , state_symbol=state_symbol), code=307)


#global vars for easy reusability
global model, graph
model, graph = init()


#preprocessing globals
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])
sc_X = StandardScaler()


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    global prec
    
    state = request.args.get('state')
    prec = float(request.args.get('precipitation'))
    month = request.args.get('month')
    terrain = request.args.get('terrain')
    year = request.args.get('year')
    actualMonth = int(request.args.get('actualMonth'))
    state_symbol = request.args.get('state_symbol')
    
    request.args.get('terrain')
    
    currentMonth = datetime.datetime.now().month
    currentYear = datetime.datetime.now().year
    
    #optional parameters
    city = request.args.get('city')
    duration = float(request.args.get('duration'))
    
    #normalization of precipitaion over the duration
    prec = prec * 30 * duration
    
    #we take average historicat rainfall 2016 onwards as rainfall data is unavailable 
    #however we use openwather api to get rainfall from current day and upto the end of current month 
    if( (int(year)==2016 or int(year)==2017 or( int(year)==2018 and actualMonth< currentMonth) ) or int(year) > currentYear or ( int(year) == currentYear and actualMonth > currentMonth)):
        RainfallData = pd.read_csv('RainfallData.csv')
        given_state = RainfallData['STATE']==state_symbol
        RainfallData = RainfallData[given_state]
        prec = RainfallData[month].iloc[0]       
    
    if( int(year) <= 2015 ):
        dataset = pd.read_csv('updated_test.csv')
        dataset = dataset[ dataset['YEAR']>1979 ]
        dataset = dataset.dropna()
        dataset = dataset.iloc[:,[0,1,3,4,6,8]]
        sd = dataset['SUBDIVISION'] == state 
        yr = dataset['YEAR'] == int(year) 
        qr = dataset['QUARTER'] == month 
        tr = dataset['TERRAIN'] == terrain
        prec = float(dataset[ sd & yr & qr & tr ]['PRECIPITATION'].iloc[0])
        response = int(dataset[ sd & yr & qr & tr ]['SEVERITY'].iloc[0])
        print("predict response: ", response)
        
        if(int(response) == 0 ):
            return render_template('response0.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='0')
        elif(int(response) == 1):
            return render_template('response1.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='1')
        elif(int(response) == 2):
            return render_template('response2.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='2')
        elif(int(response) == 3):
            return render_template('response3.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='3')
        elif(int(response) == 4):
            return render_template('response4.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='4')
        else:
            return render_template('response5.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='5')
        
    x = np.array([state,prec]);
    
    #preprocessing x
    x = np.reshape(x,(-1,2))
    x[:, 0] = labelencoder_X_1.transform(x[:, 0])
    x = onehotencoder.transform(x).toarray()
    x = x[:,1:]
    x = sc_X.transform(x)
    
    with graph.as_default():
        out = model.predict(x)
        response = np.argmax(out,axis=1)[0]
        print("predict response: ", response)
        if(int(response) == 0 ):
            return render_template('response0.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='0')
        elif(int(response) == 1):
            return render_template('response1.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='1')
        elif(int(response) == 2):
            return render_template('response2.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='2')
        elif(int(response) == 3):
            return render_template('response3.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='3')
        elif(int(response) == 4):
            return render_template('response4.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='4')
        else:
            return render_template('response5.html', city=city, state=state, month=month, duration=duration, precipitation=round(prec, 2), terrain=terrain, severity=response, year=year, level='5')

def createEncoderandScaler():
        
    global labelencoder_X_1,labelencoder_X_2,labelencoder_X_3,onehotencoder,sc_X
    
    dataset = pd.read_csv('updated_test.csv')
    dataset = dataset[ dataset['YEAR']>1980 ]
    dataset = dataset.dropna()
    #without climate
    X = dataset.iloc[:,[0,4]].values
    y = dataset.iloc[:,8].values
     
    X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
        
    X = onehotencoder.fit_transform(X).toarray()
    
    X = X[:,1:]
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
    onehotencoder_2 = OneHotEncoder(categorical_features = [0])
    y_train = np.reshape(y_train,(-1,1))
    y_train = onehotencoder_2.fit_transform(y_train).toarray()
    
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    

if __name__ == "__main__":
    createEncoderandScaler()
    app.run()
