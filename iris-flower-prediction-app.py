#import modules
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# header and description
# one hash means h1
# double asterisk means bold
st.write("""
# Simple Iris Flower Prediction App

This app predicts the type of **Iris flower**!
""")

# sidebar header
st.sidebar.header("User Input Parameters")

# set sliders
# first parameter is label, second parameter is min value
# third parameter is max value, fourth parameter is default value
sepal_length = st.sidebar.slider("Sepal Length",4.3,7.9,5.4)
sepal_width = st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
petal_length = st.sidebar.slider("Petal Length",1.0,6.9,1.3)
petal_width = st.sidebar.slider("Petal Width",0.1,2.5,0.2)

# set value into a dictionary
data = {
    'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width,
}

# set to dataframe
df = pd.DataFrame(data,index=[0])

# output DataFrame
st.subheader('User Input Parameters')
st.write(df)

# load iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
# Builds a forest of trees from X and Y
clf.fit(X,Y)

# prediction
prediction = clf.predict(df)

# predicition probability (how probable the prediction is)
prediction_probability = clf.predict_proba(df)

# output names of target
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

# output prediction
st.subheader("Prediction")
st.write(iris.target_names[prediction])

# output prediction probability
st.subheader("Prediction Probability")
st.write(prediction_probability)