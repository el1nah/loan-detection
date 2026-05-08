import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , accuracy_score , precision_score , confusion_matrix , recall_score , f1_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

'''
df=pd.read_csv('loan_detection.csv')
df.columns_to_process = ['not_working', 'job_admin.', 'job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed', 'job_services','job_student','job_technician','job_unemployed', 'job_unknown']

for col in df.columns_to_process:
    df[col] = df[col].apply(lambda x: col if x == 1 else (''))
   
df['job'] = df['not_working'] + df['job_admin.']+ df['job_blue-collar']+df['job_entrepreneur']+df['job_housemaid']+df['job_management']+df['job_retired']+df['job_self-employed']+df['job_services']+df['job_student']+df['job_technician']+df['job_unemployed']+ df['job_unknown']
df = df.drop(columns= ['not_working', 'job_admin.', 'job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed', 'job_services','job_student','job_technician','job_unemployed', 'job_unknown'])
df['job']=df['job'].replace('not_workingjob_retired', 'retired')
print(df) 
'''

df = pd.read_csv('loan_detection.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

df['Loan_Status_label'].value_counts()
for col in df.columns:
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

#checking the count for loan approved and loan rejected
print(df['Loan_Status_label'].value_counts())


#df = df.drop_duplicates()

df.shape


#checking null values
print(df.isnull().sum())

#visualisation of approved vs rejected loan applications with the help of a barplot
status = list(df.Loan_Status_label)
approved = status.count(1)
rejected = status.count(0)
labels = ['Loan Approved', 'Loan Rejected']
counts = [approved, rejected]
plt.bar(labels, counts, color=['#749da1', '#424c51'])
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Loan Status')
print(plt.show())


#correlation between other columns or features and 'Loan_Status_label'
correlation_matrix = df.corr()
print(correlation_matrix['Loan_Status_label'])


#splitting data into train and test sets
x = df.drop('Loan_Status_label', axis=1)
y = df['Loan_Status_label']
x_train, x_test, y_train, y_test= train_test_split(x , y , test_size = 0.2 , random_state = 42)

print("X_train shape:", x_train.shape) 
print("X_test shape :", x_test.shape) 
print("y_train shape:", y_train.shape) 
print("y_test shape:", y_test.shape) 


#Model Training and Prediction
lr = LogisticRegression()
lr.fit(x_train , y_train)
y_train_pred_lr = lr.predict(x_train)
y_test_pred_lr= lr.predict(x_test)

#Training Evaluation 
print(confusion_matrix(y_train, y_train_pred_lr))
print(accuracy_score(y_train, y_train_pred_lr))
print(classification_report(y_train, y_train_pred_lr))

#Plotting Heatmap for Training Data
sns.heatmap(confusion_matrix(y_train, y_train_pred_lr), annot=True)
plt.title('Confusion Matrix for Training Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(plt.show())

#Test Evaluation
print(confusion_matrix(y_test, y_test_pred_lr))
print(accuracy_score(y_test, y_test_pred_lr))
print(classification_report(y_test, y_test_pred_lr))

#Plotting Heatmap for Test Data
sns.heatmap(confusion_matrix(y_train, y_train_pred_lr), annot=True)
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(plt.show())


#SVM Model Building
svm=SVC()
svm.fit(x_train , y_train)
y_train_pred_svm = svm.predict(x_train)
y_test_pred_svm = svm.predict(x_test)

#Training
print(confusion_matrix(y_train, y_train_pred_svm))
print(accuracy_score(y_train, y_train_pred_svm))
print(classification_report(y_train, y_train_pred_svm))

#Testing
print(confusion_matrix(y_test, y_test_pred_svm))
print(accuracy_score(y_test, y_test_pred_svm))
print(classification_report(y_test, y_test_pred_svm))

#KNN Model Building
knn= KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train , y_train)
y_train_pred_knn = knn.predict(x_train)
y_test_pred_knn = knn.predict(x_test)

#Training
print(confusion_matrix(y_train, y_train_pred_knn))
print(accuracy_score(y_train, y_train_pred_knn))
print(classification_report(y_train, y_train_pred_knn))

#Testing
print(confusion_matrix(y_test, y_test_pred_knn))
print(accuracy_score(y_test, y_test_pred_knn))
print(classification_report(y_test, y_test_pred_knn))


#Decision Tree
dtree = DecisionTreeClassifier(max_depth = 4)
dtree.fit(x_train , y_train)
y_train_pred_dt = dtree.predict(x_train)
y_test_pred_dt= dtree.predict(x_test)

#Training
print(confusion_matrix(y_train, y_train_pred_dt))
print(accuracy_score(y_train, y_train_pred_dt))
print(classification_report(y_train, y_train_pred_dt))

#Testing
print(confusion_matrix(y_test, y_test_pred_dt))
print(accuracy_score(y_test, y_test_pred_dt))
print(classification_report(y_test, y_test_pred_dt))

#Plotting Tree
plt.figure(figsize = (25,6))
tree.plot_tree(dtree)
print(plt.show())

accuracy_train_lr= accuracy_score(y_train, y_train_pred_lr)
accuracy_train_svm= accuracy_score(y_train, y_train_pred_svm)
accuracy_train_knn= accuracy_score(y_train, y_train_pred_knn)
accuracy_train_dt=accuracy_score(y_train, y_train_pred_dt)

accuracy_test_lr=accuracy_score(y_test, y_test_pred_lr)
accuracy_test_svm=accuracy_score(y_test, y_test_pred_svm)
accuracy_test_knn=accuracy_score(y_test, y_test_pred_knn)
accuracy_test_dt=accuracy_score(y_test, y_test_pred_dt)
df['acctr']= 'accuracy_train_lr'+ 'accuracy_train_svm'+ 'accuracy_train_knn' + 'accuracy_train_dt'


parameters = {'criterion' : ['gini' , 'entropy' , 'log_loss'],'splitter' : ['best' , 'random'],'max_depth' : [1,2,3,4,5],'max_features' : ['sqrt' , 'auto' , 'log2']}

dtree2 = DecisionTreeClassifier(random_state = 42)
grid = GridSearchCV(dtree2 , param_grid = parameters , cv = 5 ,scoring = 'accuracy')
grid.fit(x_train , y_train)

#print(grid.best_params_)

y_train_pred_dt2 = grid.predict(x_train)
y_test_pred_dt2 = grid.predict(x_test)

#print(y_train_pred_dt2,  y_test_pred_dt2)

#Training
print(confusion_matrix(y_train, y_train_pred_dt2))
print(accuracy_score(y_train, y_train_pred_dt2))
print(classification_report(y_train, y_train_pred_dt2))

#Testing
print(confusion_matrix(y_test, y_test_pred_dt2))
print(accuracy_score(y_test, y_test_pred_dt2))
print(classification_report(y_test, y_test_pred_dt2))


