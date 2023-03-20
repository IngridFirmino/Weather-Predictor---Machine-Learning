import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime

df = pd.read_csv("final_project.csv").reset_index()
df = df.drop("Unnamed: 0",axis=1)
y = df['Solar Irradiance']
X = df[['Day_index', 'Tmod_y', "Relative Humidity"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_reg.score(X_test, y_test)

pred = xgb_reg.predict(X_test)

np.sqrt(mean_squared_error(y_test,pred))

def main_pred(humidity, date):
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    day_of_year = date_obj.timetuple().tm_yday
    
    df_tmod = pd.read_excel("T_mod.xlsx")
    df_tmod = df_tmod.drop("Time", axis=1)
    test = df_tmod[df_tmod["Day_index"] == day_of_year]
    test["Relative Humidity"] = float(humidity)
    
    data_pred = xgb_reg.predict(test)
    df1 = pd.DataFrame(data_pred, columns=['Solar_Irradiance'])
    
    df2 = test.copy()
    df2["Solar_Irradiance"] = df1["Solar_Irradiance"].values
    df3 = zero_solar_irrad(df2) # call the zero_solar_irrad function here
    
    # create final dataframe
    df_final = pd.concat([df3["Day_index"], df3["Relative Humidity"], df3["Tmod_y"], df3["Solar_Irradiance"]], axis=1)
    
    return df_final

def zero_solar_irrad(df):
    df['Solar_Irradiance'] = df.apply(lambda row: 0 if row['Tmod_y'] == 0 else row['Solar_Irradiance'], axis=1)
    return df

def concat_solar(df):
    df["Solar Irradiance 2"] = df["Solar_Irradiance"].shift(periods=1, fill_value=0).rename("Solar Irradiance 2")
    return df.reset_index(drop=True)

def compare_tmod(df):
    df["Tmod_x"] = df["Tmod_y"].max()
    df["T_mod_comp"] =df["Tmod_y"] ==df["Tmod_x"]
    return df

def solar_diference(df):
    df["Solar_diference"] = df["Solar_Irradiance"] - df["Solar Irradiance 2"]
    return df

def compare(df):
    counter = 0
    results = []
    t_mod_list = list(df["T_mod_comp"])
    index = t_mod_list.index(True)
    solar_list = list(df["Solar_diference"])
    for i in solar_list:
        if counter < index:
            results.append(i>=0)
            counter+=1
        else:
            results.append(i<=0)
            counter+=1
    df["sunny"] = results           
    return df
    

def main(df):
    df = concat_solar(df)
    df = compare_tmod(df)
    df = solar_diference(df)
    df = compare(df)

    return df

def cloudy_check(df):
    a = list(df["sunny"])
    if len(a) == sum(a):
        return "Sunny"
    else:
        return "Cloudy"

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def iris_pred():

    if request.method == 'POST':
               
        humidity = float(request.form['humidity'])
        date = request.form['date']
        df = main_pred(humidity, date)
        df1 = main(df)
        cloudy = cloudy_check(df1)
        df2 = df1[["Day_index","Relative Humidity","Solar_Irradiance","sunny"]]

        
        return render_template('index.html', tables=df2.to_html(classes='data', header="true"), cloudy = cloudy)
    else:
        return render_template('information.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)