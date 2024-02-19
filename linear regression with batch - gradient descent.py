#تمرین بخش اول سری دوم داده کاوی
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

#تمرین بخش سوم سری دوم داده کاوی
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn import  model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score , f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV
from yellowbrick.target import ClassBalance
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics, cross_validation
from sklearn.cross_validation import cross_val_score
from nltk import ConfusionMatrix
#from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
df=pd.read_csv(r'/home/pegah/Desktop/train.csv')
print(df.head())
df.columns
print(df.describe().T)
df=df.rename(columns={'blue':'bluetooth','fc':'fc_megapixel','pc':'pc_megapixel','m_dep':'m_depth'})
dupe=df.duplicated()
print(dupe.head())
sum(dupe)
df.isnull().sum()
df['fc_megapixel']=df['fc_megapixel'].fillna(0)
df['fc_megapixel']
df['four_g']=df['four_g'].fillna(df['four_g'].mean())
df['four_g']
len(df['pc_megapixel'].unique())
df['pc_megapixel']=df['pc_megapixel'].fillna(method='bfill')
len(df['pc_megapixel'].unique())
df.to_csv('mobile price ddata cleaned.csv',index=False)
#normalize numeric values and delete outlier
#for cols in mobile_train.columns:
    #if mobile_train[cols].dtype=='int64' or mobile_train[cols].dtype=='float64':
        #mobile_train[cols]=((mobile_train[cols]-mobile_train[cols].mean())/(mobile_train[cols].std()))
#mobile_train.head() 
#mobile_train.shape       
#delete outlier
#data.shape
#for cols in mobile_train.columns:
    #if mobile_train[cols].dtype=='int64' or mobile_train[cols].dtype=='float64':
        #upper_range=mobile_train[cols].mean()+3*mobile_train[cols].std()
        #lower_range=mobile_train[cols].mean()-3*mobile_train[cols].std()
        #indexs=mobile_train[(mobile_train[cols]>upper_range)| (mobile_train[cols]<lower_range)].index
        #mobile_train=mobile_train.drop(indexs)
#mobile_train.shape

numericData=df.drop(['bluetooth','dual_sim','four_g','three_g','touch_screen','wifi','price_range'],axis=1)
print(numericData.head())
categoricalData=df[['bluetooth','dual_sim','four_g','three_g','touch_screen','wifi','price_range']]
print(categoricalData.head())
fig , ax=plt.subplots(figsize=(10,5))
sns.boxplot(numericData['ram'],orient='v')
msno.bar(df)
plt.show()
sns.pairplot(data=df , hue='price_range')
plt.show()
#ram affected by price
sns.jointplot(x='ram',y='price_range',data=df , color='red',kind='kde')
plt.show()
sns.set(rc={'figure.figsize':(10,7)})
sns.stripplot(x="price_range",y="ram",data=df,dodge=True,palette='dark')
plt.show()
sns.scatterplot(x="ram" ,y="battery_power" , hue="price_range" , data=df , palette='deep')
plt.show()

df['price_range'].value_counts()

plt.hist(df["price_range"],bins=4)
df.loc[df["price_range"]==0,'price_range'].count()
df.head()
sns.boxplot(data=df,x='price_range',y='sc_w')
sns.boxplot(data=df,x='price_range',y='battery_power')
scaler=StandardScaler()
scale_array=scaler.fit_transform(numericData)
scaleData=pd.DataFrame(scale_array,columns=numericData.columns)
print(scaleData.head())
scaleData.describe()
fig,ax=plt.subplots(figsize=(10,5))
all=sns.boxplot(data=scaleData)
all.set_xticklabels(all.get_xticklabels(),rotation=90)
Q1=numericData.quantile(0.25)
Q3=numericData.quantile(0.75)
IQR=Q3-Q1
print(IQR)
removeOutlierData=numericData[~((numericData<(Q1-1.5*IQR))\
                                |(numericData>(Q3+1.5*IQR))).any(axis=1)]
removeOutlierData.shape

fig, ax=plt.subplots(figsize=(10,5))
all=sns.boxplot(data=removeOutlierData)
all.set_xticklabels(all.get_xticklabels(),rotation=90)
corr=df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,cmap="RdYlGn")
corr_matrix=corr
corr_matrix['price_range']
sns.countplot(df['price_range'])
plt.show()

scaleData=scaleData.reset_index()
print(scaleData.head())
categoricalData=categoricalData.reset_index()
print(categoricalData.head())
finalDf=pd.concat([scaleData , categoricalData],axis=1)
print(finalDf.head())
X=finalDf.drop('price_range',axis=1)
Y=finalDf['price_range']
scaler.fit(X)
x=scaler.transform(X)
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=101)
x_train.shape
x_test.shape
#logistic regression Q1
logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=100000)
logreg.fit(x_train, Y_train)
Y_pred = logreg.predict(x_test)
z=logreg.score(x_test, Y_test)
print(z)
log=accuracy_score(Y_pred,Y_test)*100
print(log)
cm_log=confusion_matrix(Y_pred,Y_test)
print(cm_log)
cl=classification_report(Y_pred,Y_test)
print(cl)
#Q2 balanced class
vi=ClassBalance(labels=[0,1,2,3])
vi.fit(Y_train,Y_test)
vi.poof()
#Q3 new class
Y_trainn=Y_train.map({0:0,1:1,2:1,3:1})
Y_testn=Y_test.map({0:0,1:1,2:1,3:1})
#Q4 logistic regression with new labels
logregn = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=100000)
logregn.fit(x_train, Y_trainn)
Y_predn = logregn.predict(x_test)
z=logregn.score(x_test, Y_testn)
print(z)
logre=accuracy_score(Y_predn,Y_testn)*100
print(logre)
cm_logre=confusion_matrix(Y_predn,Y_testn)
print(cm_logre)
cln=classification_report(Y_predn,Y_testn)
print(cln)
#Q5 logistic regression for imbalenced data
rus=RandomUnderSampler(random_state=42,replacement=True)
x_rus,y_rus=rus.fit_resample(x_train,Y_trainn)
print('original data shape:',Counter(Y_trainn))
print('resample data shape:',Counter(y_rus))
x_rus_test,y_rus_test=rus.fit_resample(x_test,Y_testn)
print('original data shape:',Counter(Y_testn))
print('original data shape:',Counter(y_rus_test))
logregnn = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=100000)
logregnn.fit(x_rus, y_rus)
Y_prednn = logregnn.predict(x_rus_test)
z=logregnn.score(x_rus_test, y_rus_test)
print(z)
logren=accuracy_score(Y_prednn,y_rus_test)*100
print(logren)
cm_logren=confusion_matrix(Y_prednn,y_rus_test)
print(cm_logren)
clnn=classification_report(Y_prednn,y_rus_test)
print(clnn)


#Q11 cross validation for logistic
features_col=['battery_power','bluetooth','clock_speed','dual_sim','fc_megapixel','four_g','int_memory','m_depth','mobile_wt','n_cores','pc_megapixel','px_height','px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi']
X=df[features_col]
y=df['price_range']
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y, test_size=0.2, random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train_n,y_train_n)
predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)
metrics.accuracy_score(y, predicted) 
accuracy = cross_val_score(logreg, X, y, cv=10,scoring='accuracy')
print (accuracy)
print (cross_val_score(logreg, X, y, cv=10,scoring='accuracy').mean())
print (ConfusionMatrix(list(y), list(predicted)))
print (metrics.recall_score(y, predicted) )
probs = logreg.predict_proba(X)[:, 1] 
plt.hist(probs) 
plt.show()
preds = np.where(probs > 0.5, 1, 0) 
print (ConfusionMatrix(list(y), list(preds)))
print (metrics.accuracy_score(y, predicted)) 
fpr, tpr, thresholds = metrics.roc_curve(y, probs) 
plt.plot(fpr, tpr) 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.0]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate)') 
plt.show()

# calculate AUC 
print (metrics.roc_auc_score(y, probs))

# use AUC as evaluation metric for cross-validation 
logreg = LogisticRegression() 
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean() 

