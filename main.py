#importing necessary libraries
import pandas as pd
import pickle

#reading the csv file
df = pd.read_csv('Iris.csv')

#splitting input and output
x = df.iloc[:,[1,2,3,4]].values
y = df['Species'].values

#splitting training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#fitting the model
from sklearn.svm import SVC
model = SVC(kernel='linear').fit(x_train,y_train)

#dumping the model into a pickle file
pickle.dump(model,open('iris.pkl','wb'))