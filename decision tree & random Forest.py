import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import boxcox
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error 
#from optbinning import BinningProcess
#from sklearn.pipeline import Pipeline
#processing on datas
data_train=pd.read_csv(r'/home/pegah/Desktop/train.csv')
print(data_train.head(7))
data_test=pd.read_csv(r'/home/pegah/Desktop/test.csv')
print(data_test.head(6))
data_test=data_test.drop('id',axis=1)
print(data_train.shape)
print(data_test.shape)
data_train.info()
data_test.info()
data_train.describe()
data_train.isnull()
data_train.isnull().sum()
data_test.isnull().sum()
data_train.plot(x='price_range',y='ram',kind='scatter')
data_train.plot(x='price_range',y='fc',kind='scatter')
data_train.plot(x='price_range',y='clock_speed',kind='scatter')
data_train.plot(kind='box',figsize=(20,10))
plt.show()
X=data_train.drop('price_range',axis=1)
print(X)
Y=data_train['price_range']
print(Y)
#std=StandardScaler()
#X_std=std.fit_transform(X)
#data_test_std=std.transform(data_test)
#print(X_std)
#print(data_test_std)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
svc1=SVC(kernel='linear',random_state=0)
svc1.fit(x_train,y_train)
y_pred1=svc1.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
print('confusion matrix:\n' , cm1)
sva1=accuracy_score(y_test,y_pred1)
print('accuracy score for kernel linear= ', accuracy_score(y_test,y_pred1))
valid_rmse=np.sqrt(mean_squared_error(y_test,y_pred1))
print(valid_rmse)

svc2=SVC(kernel='rbf',random_state=0)
svc2.fit(x_train,y_train)
y_pred2=svc2.predict(x_test)
cm2=confusion_matrix(y_test,y_pred2)
print('confusion matrix:\n' , cm2)
sva2=accuracy_score(y_test,y_pred2)
print('accuracy score for kernel RBF= ', accuracy_score(y_test,y_pred2))

svc3=SVC(kernel='poly',random_state=0)
svc3.fit(x_train,y_train)
y_pred3=svc3.predict(x_test)
cm3=confusion_matrix(y_test,y_pred3)
print('confusion matrix:\n' , cm3)
sva3=accuracy_score(y_test,y_pred3)
print('accuracy score for kernel Poly= ', accuracy_score(y_test,y_pred3))

svc4=SVC(kernel='sigmoid',random_state=0)
svc4.fit(x_train,y_train)
y_pred4=svc4.predict(x_test)
cm4=confusion_matrix(y_test,y_pred4)
print('confusion matrix:\n' , cm4)
sva4=accuracy_score(y_test,y_pred4)
print('accuracy score for kernel Sigmoid= ', accuracy_score(y_test,y_pred4))

K = chi2_kernel(x_train, gamma=0.5)
svc5 = SVC(kernel=chi2_kernel).fit(x_train, y_train)
y_pred5=svc5.predict(x_test)
cm5=confusion_matrix(y_test,y_pred5)
print('confusion matrix:\n' , cm5)
sva5=accuracy_score(y_test,y_pred5)
print('accuracy score for kernel precomputed= ', accuracy_score(y_test,y_pred5))
#parameters=[{'C':[0.25,0.5,0.75,1],'kernel':['linear']},{'C':[0.25,0.5,0.75,1],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},{'C':[0.25,0.5,0.75,1],'kernel':['poly']},{'C':[0.25,0.5,0.75,1],'kernel':['sigmoid']}]
#grid_search=GridSearchCV(estimator=svc1,param_grid=parameters,scoring='accuracy',cv=10,n_job=-1)
#grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#best_parameters=grid_search.best_params_
#print("best accuracy:{:.2f}%".format(best_accuracy*100))
#print("best parameters:",best_parameters)
#soft margin and hard margin
cf=SVC(kernel='linear',C=0.01)
cf.fit(x_train,y_train)
pre=cf.predict(x_test)
c=confusion_matrix(y_test,pre)
print('confusion matrix:\n' , c)
sva=accuracy_score(y_test,pre)
print('accuracy score for soft margin= ', accuracy_score(y_test,pre))
cf=SVC(kernel='linear',C=0.2)
cf.fit(x_train,y_train)
pre=cf.predict(x_test)
c=confusion_matrix(y_test,pre)
print('confusion matrix:\n' , c)
sva=accuracy_score(y_test,pre)
print('accuracy score for hard margin= ', accuracy_score(y_test,pre))
#binning for battery_power
x=data_train.drop('battery_power',axis=1)
print(x)
y=data_train['battery_power']
print(y)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='quantile')
est.fit(x)
Xt = est.transform(x)
print(Xt)
est.bin_edges_[3]
p=est.inverse_transform(Xt)
print(p)
#one hot encoding for categorical features
#X=data_train.drop('price_range',axis=1)
#le = preprocessing.LabelEncoder()
#X_2 = X.apply(le.fit_transform)
#enc = preprocessing.OneHotEncoder()
#enc.fit(X_2)
#onehotlabels = enc.transform(X_2).toarray()
#onehotlabels.shape
#print(onehotlabels)
data = ['touch_screen', 'wifi', 'three_g', 'dual_sim', 'four_g', 'blue']
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
#log transformation
transformer = FunctionTransformer(np.log1p, validate=True)
X1=data_train.drop('price_range',axis=1)
#print(X)
Y1=data_train['price_range']
#print(Y)
aa1=transformer.transform(X1)
print(aa1)
#another transformation
pt = PowerTransformer()
print(pt.fit(X1))
print(pt.lambdas_)
print(pt.transform(X1))
#new feature
sLength = len(data_train['price_range'])
data_train.loc[:,'volume'] = pd.Series(np.random.randn(sLength), index=data_train.index)
print(data_train.head(7))

#SVM 
#normalize numeric values and delete outlier
#for cols in data_train.columns:
 #   if data_train[cols].dtype=='int64' or data_train[cols].dtype=='float64':
  #      data_train[cols]=((data_train[cols]-data_train[cols].mean())/(data_train[cols].std()))
#data_train.head(7) 
#mobile_train.shape 
x1=data_train.drop('battery_power',axis=1)
#print(x)
y1=data_train['battery_power']
scaler=StandardScaler()
scaler.fit(x1)
x2=scaler.transform(x1)
est_train_x, est_test_x, est_train_y, est_test_y = train_test_split(x1, y1, test_size = 0.2, random_state=0)
#est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='quantile')
#est.fit(x1)
#est_train_x,est_test_x,est_train_y,est_test_y=train_test_split(y1,x1,test_size=0.2,random_state=0)
svc6=SVC(kernel='linear',random_state=0)
svc6.fit(est_train_x,est_train_y)
y_pred6=svc6.predict(est_test_x)
cm6=confusion_matrix(est_test_y,y_pred6)
print('confusion matrix:\n' , cm6)
sva6=accuracy_score(est_test_y,y_pred6)
print('accuracy score for kernel linear in binning method= ', accuracy_score(est_test_y,y_pred6))

y2=data_train['price_range']
est_train_x, est_test_x, est_train_y, est_test_y = train_test_split(aa1, y2, test_size = 0.2, random_state=0)
svc7=SVC(kernel='rbf',random_state=0)
svc7.fit(est_train_x,est_train_y)
y_pred7=svc7.predict(est_test_x)
cm7=confusion_matrix(est_test_y,y_pred7)
print('confusion matrix:\n' , cm7)
sva6=accuracy_score(est_test_y,y_pred7)
print('accuracy score for kernel linear in log transform method= ', accuracy_score(est_test_y,y_pred6))

#x2=data_train.drop('volume',axis=1)
#y3=data_train['volume']
#est_train_x, est_test_x, est_train_y, est_test_y = train_test_split(x2, y3, test_size = 0.2, random_state=0)
#svc7=SVC(kernel='linear',random_state=0)
#svc7.fit(est_train_x,est_train_y)
#y_pred7=svc7.predict(est_test_x)
#cm7=confusion_matrix(est_test_y,y_pred7)
#print('confusion matrix:\n' , cm7)
#sva6=accuracy_score(est_test_y,y_pred7)
#print('accuracy score for kernel linear for new feature= ', accuracy_score(est_test_y,y_pred6))

#decision tree
dt=tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
tree.plot_tree(dt)
y_pred=dt.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix:\n',cm)
dta=accuracy_score(y_test,y_pred)
print('accuracy score = ',accuracy_score(y_test,y_pred))
#other parameters in decision tree
#dt = DecisionTreeClassifier()
#dt.fit(x_train, y_train)
#dt1=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,max_features=None, max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,random_state=None, splitter='best')
#y_pred1 = dt.predict(x_test)
#cm1=confusion_matrix(y_test,y_pred1)
#print(' other parameters confusion matrix:',cm1)
#dta1=accuracy_score(y_test,y_pred1)
#print(' other parameters accuracy score = ',accuracy_score(y_test,y_pred1))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt1=DecisionTreeClassifier(class_weight='balance', criterion='entropy', max_depth=10,max_features='log2', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,random_state=None, splitter='random')
y_pred1 = dt.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
print(' other parameters confusion matrix:',cm1)
dta1=accuracy_score(y_test,y_pred1)
print(' other parameters accuracy score = ',accuracy_score(y_test,y_pred1))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt1=DecisionTreeClassifier(class_weight='balance', criterion='entropy', max_depth=20,max_features='sqrt', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,random_state=None, splitter='random')
y_pred1 = dt.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
print(' other parameters sqrt confusion matrix:',cm1)
dta1=accuracy_score(y_test,y_pred1)
print(' other parameters sqrt accuracy score = ',accuracy_score(y_test,y_pred1))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt1=DecisionTreeClassifier(class_weight='balance', criterion='entropy', max_depth=20,max_features='log2', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=4,min_samples_split=2, min_weight_fraction_leaf=0.0,random_state=None, splitter='random')
y_pred1 = dt.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
print(' other parameters for changing the depth and features confusion matrix:',cm1)
dta1=accuracy_score(y_test,y_pred1)
print(' other parameters for changing the depth and features accuracy score = ',accuracy_score(y_test,y_pred1))

#pruning tree
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1])
plt.xlabel("effective alpha")
plt.ylabel("total depth")

acc_scores = [accuracy_score(y_test, clf.predict(x_test)) for clf in clfs]

tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.grid()
plt.plot(ccp_alphas[:-1], acc_scores[:-1])
plt.xlabel("effective alpha")
plt.ylabel("Accuracy scores")  
#random forest
rf=RandomForestClassifier(criterion='entropy',random_state=0)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(' random forest confusion matrix:\n',cm)
rfa=accuracy_score(y_test,y_pred)
print(' random forest accuracy score =',accuracy_score(y_test,y_pred))
valid_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(valid_rmse)