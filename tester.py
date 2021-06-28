import numpy as np
#importing numpy for numpy_array
import pandas as pd
#Importing Pandas for reading CSV files
import matplotlib.pyplot as plt
#importing matplotlib for plotting graphs 
import seaborn as sns
#importing seaborn for plotting heatmap
from sklearn.model_selection import train_test_split
#importing sklearn for regression method
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,export_text,export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
     
df=pd.read_csv("uber-raw-data-apr14.csv","r")
#df=df.reset_index()

#header for the dataset working with
print(df.head(5), end="\n *********header above********* \n")
# duplicates and null removal
df.drop_duplicates()
df.dropna()

# describe
print(df.describe(include=['object','number']), end=" \n**descriptitive table above**\n")

#nunique function
print("type of data: ",str(type(df)))
print(df.nunique(), end="\n**nunique values**\n")

#column names of the dataframe
for i in df.columns:
    print(i, end="**\n")
print(df.shape)
print(df.keys())
print(df.dtypes)

plt.figure(figsize=[15,21])
for i in range(1,df.shape[1]):
    column = df.columns[i]
    plt.subplot(2,1,i)
    plt.hist(df[column])