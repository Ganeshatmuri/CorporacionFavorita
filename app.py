from flask import Flask, request,render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
crop_recommendation_model_path = 'final.pkl'
model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
#df=pd.read_csv("kaggle_train.csv")
#X=df.drop("sales",1)
#y=df.sales.values.reshape(-1,1)
#X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)
#yscaler=MinMaxScaler()
#yscaler.fit(y_train)
#y_train=yscaler.transform(y_train)
@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict",methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method=='POST':
        temp=list()
        #item bought date
        date=request.form['Date']
        day=int(pd.to_datetime(date,format="%Y-%m-%d").day)
        month=int(pd.to_datetime(date,format="%Y-%m-%d").month)
        year=int(pd.to_datetime(date,format="%Y-%m-%d").year)
        
        state=request.form['city']
        if state=='c_Quito':
            temp=temp+[1,0,0,0,0,0,0,0,0,0]
        elif state=='c_Cayambe':
            temp=temp+[0,1,0,0,0,0,0,0,0,0]
        elif state=='c_Cuenca':
            temp=temp+[0,0,1,0,0,0,0,0,0,0]
        elif state=='c_Ambato':
            temp=temp+[0,0,0,1,0,0,0,0,0,0]
        elif state=='c_Daule':
            temp=temp+[0,0,0,0,1,0,0,0,0,0]
        elif state=='c_Loja':
            temp=temp+[0,0,0,0,0,1,0,0,0,0]
        elif state=='c_Manta':
            temp=temp+[0,0,0,0,0,0,1,0,0,0]
        elif state=='c_Babahoyo':
            temp=temp+[0,0,0,0,0,0,0,1,0,0]
        elif state=='c_Libertad':
            temp=temp+[0,0,0,0,0,0,0,0,1,0]
        elif state=='c_Esmeraldas':
            temp=temp+[0,0,0,0,0,0,0,0,0,1]

        store=request.form['store_nbr']
        if store=='1':
            store=1
        elif store=='2':
            store=2
        elif store=='3':
           store=3
        elif store=='4':
           store=4
        elif store=='6':
            store=6
        elif store=='7':
            store=7
        elif store=='8':
            store=8
        elif store=='9':
            store=9
        elif store=='10':
            store=10
        elif store=='17':
            store=17
        elif store=='18':
            store=18
        elif store=='20':
            store=20
        elif store=='44':
            store=44
        elif store=='45':
            store=45
        elif store=='46':
            store=46
        elif store=='47':
            store=47
        elif store=='48':
            store=48
        elif store=='49':
            store=49
        elif store=='11':
            store=11
        elif store=='37':
            store=37
        elif store=='39':
            store=39
        elif store=='42':
            store=42
        elif store=='50':
            store=50
        elif store=='27':
            store=27
        elif store=='38':
            store=38
        elif store=='52':
            store=52
        elif store=='53':
            store=53
        elif store=='31':
            store=31
        elif store=='36':
            store=36
        elif store=='43':
            store=43
        
        
                
        #family type
        temp1=list()
        family=request.form['family']
        if family=='GROCERY I':
            temp1=temp1+[1,0,0,0,0,0,0,0,0,0]
        elif family=='BEVERAGES':
            temp1=temp1+[0,1,0,0,0,0,0,0,0,0]
        elif family=='PRODUCE':
            temp1=temp1+[0,0,1,0,0,0,0,0,0,0]
        elif family=='CLEANING':
            temp1=temp1+[0,0,0,1,0,0,0,0,0,0]
        elif family=='DAIRY':
            temp1=temp1+[0,0,0,0,1,0,0,0,0,0]
        elif family=='BREAD/BAKERY':
            temp1=temp1+[0,0,0,0,0,1,0,0,0,0]
        elif family=='MEATS':
            temp1=temp1+[0,0,0,0,0,0,1,0,0,0]
        elif family=='POULTRY':
            temp1=temp1+[0,0,0,0,0,0,0,1,0,0]
        elif family=='DELI':
            temp1=temp1[0,0,0,0,0,0,0,0,1,0]
        elif family=='PERSONAL CARE':
            temp1=temp1+[0,0,0,0,0,0,0,0,0,1]

        #store number
        
        temp=[day,month,year]+temp+[store]+temp1
        print(temp)
        data=np.array([temp])
        my_pred=model.predict(data)[0]
        print(my_pred)
        maximum_y=124717
        minimum_y=0
        pred=my_pred*(maximum_y-minimum_y)+minimum_y
#        pred=int(yscaler.inverse_transform(my_pred.reshape(-1,1)))
        #print(pred)
        #print(data)
        if pred<0:
            pred=0
        else:
            pred=int(pred)
        return render_template('result.html',lower_limit=abs(pred-10),upper_limit=abs(pred+10))
    
if __name__=='__main__':
    app.run(debug=True)
    
