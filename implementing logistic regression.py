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
