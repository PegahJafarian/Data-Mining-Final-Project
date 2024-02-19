#تمرینات بخش اول سری دوم داده کاوی
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import zscore
from scipy import stats
import seaborn as sns
import sklearn.linear_model as skl_lm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#pd.set_option('display.max_columns',None)
data = pd.read_csv(r'/home/pegah/Desktop/livingspace.csv')
data.head()
data.info()
round(data.isna().sum() * 100 / data.shape[0] , 2)#percentage for nall values
#delete columns with more than 50% null data
data.isna().sum()/len(data) 
data.isna() #where we have null values
data.columns[((data.isna().sum()/len(data))>0.50)]
data=data.drop(columns=data.columns[((data.isna().sum()/len(data))>0.50)]) #drop columns
data.columns
#drop meaningless data
data.drop(labels = ['scoutId'],axis = 'columns' , inplace=True)
#fillna numeric data 
#data._get_numeric_data().mean() 
data.fillna(data._get_numeric_data().mean(),inplace=True)
round(data.isna().sum() * 100 / data.shape[0] , 2)
data.isna().sum()
#normalize numeric values and delete outlier
#for cols in data.columns:
    #if data[cols].dtype=='int64' or data[cols].dtype=='float64':
        #data[cols]=((data[cols]-data[cols].mean())/(data[cols].std()))
#data.head() 
#data.shape       
#delete outlier
#data.shape
for cols in data.columns:
    if data[cols].dtype=='int64' or data[cols].dtype=='float64':
        upper_range=data[cols].mean()+3*data[cols].std()
        lower_range=data[cols].mean()-3*data[cols].std()
        indexs=data[(data[cols]>upper_range)| (data[cols]<lower_range)].index
        data=data.drop(indexs)
data.shape
plt.figure(figsize=(28,8))
sns.countplot(data['heatingType'])
data.loc[:,'heatingType'].fillna('central_heating',inplace=True)

#fillna categorical data
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool':
        data[cols].fillna(data[cols].value_counts().head(1).index[0],inplace=True)
        #print('column: ' ,cols)
        #print(data[cols].value_counts().head(1).index[0])
        #print('cols:{},value:{}'.format(cols,data[cols].value_counts().head(1).index[0]))
        #print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))
#round(data.isna().sum() * 100 / data.shape[0] , 2)        
#categorical feature
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool':
        print('cols : {}, unique values : {}'.format(cols,data[cols].nunique()))
        
for cols in data.columns:
    if data[cols].dtype=='object' or data[cols].dtype=='bool': 
        print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))

data.drop(labels = ['description','houseNumber','geo_bln','geo_krs','street','facilities','regio1','regio2','regio3','streetPlain'],axis = 'columns' , inplace=True)
others = list(data['heatingType'].value_counts().tail(12).index)
def edit_heatingType(x):
    if x in others:
        return 'other'
    else:
        return x

data['heatingType_edit'] = data['heatingType'].apply(edit_heatingType)
data = data.drop(columns = ['heatingType'])
data['heatingType_edit'].value_counts()*100 / len(data)

others = list(data['telekomTvOffer'].value_counts().tail(2).index)
def edit_telekomTvOffer(x):
    if x in others:
        return 'other'
    else:
        return x

data['telekomTvOffer_edit'] = data['telekomTvOffer'].apply(edit_telekomTvOffer)
data = data.drop(columns = ['telekomTvOffer'])
data['telekomTvOffer_edit'].value_counts()*100 / len(data)

others = list(data['firingTypes'].value_counts().tail(80).index)
def edit_firingTypes(x):
    if x in others:
        return 'other'
    else:
        return x

data['firingTypes_edit'] = data['firingTypes'].apply(edit_firingTypes)
data = data.drop(columns = ['firingTypes'])
data['firingTypes_edit'].value_counts()*100 / len(data)

others = list(data['condition'].value_counts().tail(7).index)
def edit_condition(x):
    if x in others:
        return 'other'
    else:
        return x

data['condition_edit'] = data['condition'].apply(edit_condition)
data = data.drop(columns = ['condition'])
data['condition_edit'].value_counts()*100 / len(data)

others = list(data['interiorQual'].value_counts().tail(3).index)
def edit_interiorQual(x):
    if x in others:
        return 'other'
    else:
        return x

data['interiorQual_edit'] = data['interiorQual'].apply(edit_interiorQual)
data = data.drop(columns = ['interiorQual'])
data['interiorQual_edit'].value_counts()*100 / len(data)

others = list(data['typeOfFlat'].value_counts().tail(9).index)
def edit_typeOfFlat(x):
    if x in others:
        return 'other'
    else:
        return x

data['typeOfFlat_edit'] = data['typeOfFlat'].apply(edit_typeOfFlat)
data = data.drop(columns = ['typeOfFlat'])
data['typeOfFlat_edit'].value_counts()*100 / len(data)

for cols in data.columns:
    if data[cols].dtype == 'object' or data[cols].dtype == 'bool':
        print('cols : {} ,\n {}'.format(cols,data[cols].value_counts()))

#categoricalcolumns
categoricalColumns = []
for cols in data.columns:
    if data[cols].dtype == 'object' or data[cols].dtype == 'bool':
        categoricalColumns.append(cols)
#categoricalcolumns  
dummies_feature = pd.get_dummies(data[categoricalColumns])
dummies_feature.head()
for item in dummies_feature.columns:
    if dummies_feature[item].dtype == 'bool':
      dummies_feature[item+'_edit'] = dummies_feature[item].astype(int)
dummies_feature.head()
data = pd.concat([data, dummies_feature], axis=1)
#data.head()
data = data.drop(columns=categoricalColumns)
data.head()
data.shape        
#correlation matrix
corr=data.corr()
f,ax=plt.subplots(figsize=(60,60))
sns.heatmap(corr,square=True , annot=True)
#plt.xticks(range(len(corr.columns)), corr.columns);
#plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

#Correlation with output variable
cor_target = corr["livingSpace"]
#Select highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
#Correlation with output variable
cor_target = corr["livingSpace"]
#Select highly correlated features
relevant_features = cor_target[cor_target< 0]
relevant_features

#regression for most correlation with packages
y=data['livingSpace']
x1=data[['livingSpaceRange']]
s=cross_val_score(LinearRegression(),x1,y, scoring="neg_mean_squared_error",cv=5)
print("cross validated data:" , s)
predictions1=cross_val_predict(LinearRegression(),x1,y,cv=5)
print(predictions1)
plt.scatter(y,predictions1)
accuracy=metrics.r2_score(y,predictions1)
print("cross_predicted accuracy:" , accuracy)

t=cross_val_score(LinearRegression(),x1,y,scoring="neg_mean_squared_error", cv=10)
print("cross validated data:" , t)
predictions2=cross_val_predict(LinearRegression(),x1,y,cv=10)
print(predictions2)
plt.scatter(y,predictions2)
accuracy=metrics.r2_score(y,predictions2)
print("cross_predicted accuracy:" , accuracy)
temp=pd.DataFrame({'test' :y,'pred':predictions2})
print(temp.head())
temp['upper_range']=temp['test']*1.3
temp['lower_range']=temp['test']*0.6
z=temp[(temp['upper_range']>=temp['pred'])&(temp['pred']>=temp['lower_range'])].shape[0]*100/temp.shape[0]
print(z)
#regresion for 4 features with packages
y=data['livingSpace']
x2=data[['livingSpaceRange','noRooms','telekomUploadSpeed','floor']]
c=cross_val_score(LinearRegression(),x2,y, scoring="neg_mean_squared_error",cv=5)
print("cross validated data:" , c)
predictions3=cross_val_predict(LinearRegression(),x2,y,cv=5)
print(predictions3)
plt.scatter(y,predictions3)
accuracy=metrics.r2_score(y,predictions3)
print("cross_predicted accuracy:" , accuracy)

d=cross_val_score(LinearRegression(),x2,y, scoring="neg_mean_squared_error",cv=10)
print("cross validated data:" , d)
predictions4=cross_val_predict(LinearRegression(),x2,y,cv=10)
print(predictions4)
plt.scatter(y,predictions4)
accuracy=metrics.r2_score(y,predictions4)
print("cross_predicted accuracy:" , accuracy)
temp=pd.DataFrame({'test' :y,'pred':predictions4})
print(temp.head())
temp['upper_range']=temp['test']*1.3
temp['lower_range']=temp['test']*0.6
e=temp[(temp['upper_range']>=temp['pred'])&(temp['pred']>=temp['lower_range'])].shape[0]*100/temp.shape[0]
print(e)
#regression for features and target
y=data['livingSpace']
x3=data[['noRoomsRange','geo_plz','totalRent','yearConstructed','pricetrend']]
h=cross_val_score(LinearRegression(),x3,y, scoring="neg_mean_squared_error",cv=5)
print("mse linear for cv=5:" , h)
#print(scoring)
predictions5=cross_val_predict(LinearRegression(),x3,y,cv=5)
print(predictions5)
plt.scatter(y,predictions5)
accuracy=metrics.r2_score(y,predictions5)
print("cross_predicted accuracy for cv=5:" , accuracy)

i=cross_val_score(LinearRegression(),x3,y, scoring="neg_mean_squared_error",cv=10)
print("mse linear for cv=10:" , i)
predictions6=cross_val_predict(LinearRegression(),x3,y,cv=10)
print(predictions6)
plt.scatter(y,predictions6)
accuracy=metrics.r2_score(y,predictions6)
print("cross_predicted accuracy for cv=10:" , accuracy)
temp=pd.DataFrame({'test' :y,'pred':predictions6})
print(temp.head())
temp['upper_range']=temp['test']*1.3
temp['lower_range']=temp['test']*0.6
j=temp[(temp['upper_range']>=temp['pred'])&(temp['pred']>=temp['lower_range'])].shape[0]*100/temp.shape[0]
print(j)
#Ridge regression with packages
ridge=linear_model.Ridge()
y=data['livingSpace']
x4=data[['noRoomsRange','geo_plz','totalRent','yearConstructed','pricetrend']]
MSE_ridge=cross_val_score(ridge,x4,y,scoring="neg_mean_squared_error" , cv=5)
print("mse for ridge regression for cv=5:" ,MSE_ridge)
predictions=cross_val_predict(ridge,x4,y,cv=5)
print("predicted accuracy for cv=5:" ,predictions)
temp=pd.DataFrame({'test':y , 'pred':predictions})
print(temp.head())
MSE_ridge1=cross_val_score(ridge,x4,y,scoring="neg_mean_squared_error" , cv=10)
print("mse for ridge regression for cv=10:" ,MSE_ridge1)
prediction=cross_val_predict(ridge,x4,y,cv=10)
print("predicted accuracy for cv=10:",prediction)
tm=pd.DataFrame({'test':y , 'pred':prediction})
print(tm.head())
tm['upper_range']=tm['test']*1.3
tm['lower_range']=tm['test']*0.6
k=tm[(tm['upper_range']>=tm['pred'])&(tm['pred']>=tm['lower_range'])].shape[0]*100/tm.shape[0]
print(k)
#lasso regression with pakhages
lasso=linear_model.Lasso()
MSE_lasso=cross_val_score(lasso,x4,y,scoring="neg_mean_squared_error",cv=5)
print("mse lasso for cv=5:",MSE_lasso)
pred1=cross_val_predict(lasso,x4,y,cv=5)
print("predicted accuracy for cv=5:",pred1)
tem=pd.DataFrame({'test':y , 'pred':pred1})
print(tem.head())
MSE_lasso=cross_val_score(lasso,x4,y,scoring="neg_mean_squared_error",cv=10)
print("mse lasso for cv=10:",MSE_lasso)
pred2=cross_val_predict(lasso,x4,y,cv=10)
print("predicted accuracy for cv=10:",pred2)
tem=pd.DataFrame({'test':y , 'pred':pred2})
print(tem.head())
tem['upper_range']=tem['test']*1.3
tem['lower_range']=tem['test']*0.6
q=tem[(tem['upper_range']>=tem['pred'])&(tem['pred']>=tem['lower_range'])].shape[0]*100/tem.shape[0]
print(q)


