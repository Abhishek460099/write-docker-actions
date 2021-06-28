#import numpy as np
#from datetime import datetime
#import matplotlib.pyplot as plt
#x=(1,1,2,3)
#print("hello world!!" , x)

import numpy as np  #importing numpy for numpy_array
import pandas as pd  #Importing Pandas for reading CSV files
import matplotlib.pyplot as plt  #importing matplotlib for plotting graphs 
import seaborn as sns  #importing seaborn for plotting heatmap
#pip install seaborn
from sklearn.model_selection import train_test_split  #importing sklearn for regression method
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,export_text,export_graphviz
#import pydotplus
from IPython.display import Image
from sklearn.metrics import accuracy_score, confusion_matrix
 #!pip install mlxtend --upgrade --no-deps
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
 #%matplotlib inline
     
df=pd.read_csv("data_by_genres.csv","r")
 
df = df.dropna()
print(df.head())
 
print(df.isna())
df=  df.fillna(0)
print(str(df.keys()))
 
print(df.describe(percentiles=[0.35,0.65,.95]))
print(df.describe())
print(df.dtypes)
plt.figure(figsize=[15,21])
for i in range(1,df.shape[1]):
    plt.subplots(5,4)
    plt.hist(df.columns[i])
 
danceable=list()
column=df['danceability']
for index, item in enumerate(column.values):
   if(item>=0.0 and item <=0.25): danceable.append('least')
   if(item > 0.25 and item <= 0.50): danceable.append('medium')
   if(item > 0.5 and item <= 0.75): danceable.append('medium -high')
   else: danceable.append('high')
df['danceability']= danceable



