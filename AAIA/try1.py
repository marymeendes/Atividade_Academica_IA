# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
names = ['wife\'s-age', 'wife\'-education', 'husband\'s-education', 'numbers-of-children', 'wife\'s-religion', 'wife\'s-now-working', 'husband\'s-occupation', 'standard-of-living', 'media-exposure', 'cmu-class']
dataset = pandas.read_csv(url, names=names)

# shape
#print(dataset.shape)

# head
# print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('cmu-class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(10,10), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

data = [] 
target = []

    for line in dataset: 
        v = line.strip().split(",")
        data.append(v[:-1])
        target.append(v[-1])
data = np.array(data)
data = data.astype(np.float)
target = np.array(target)
target = target.astype(np.float)
print(data.shape,target.shape)

