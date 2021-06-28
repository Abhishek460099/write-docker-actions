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
df=df.reset_index()

#header
print(df.head(5))
# duplicates and null removal
df.drop_duplicates()
df.dropna()


# describe
print(df.describe(include=['object','int']))

#nunique
print(df.nunique())

#value counts
for i in df.columns:
    print(i)

