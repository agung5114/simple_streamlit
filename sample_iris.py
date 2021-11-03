import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import sklearn

# load model 
import joblib

def main():
    """App with Streamlit"""
    #title
    st.title("Hello Data Analyst!")
    st.subheader("Iris flower Prediction from Machine Learning Model")
    iris= Image.open('iris.png')
    st.image(iris)

    #load model
    model= open("model.pkl", "rb")
    knn_clf=joblib.load(model)

    #load images
    setosa= Image.open('setosa.png')
    versicolor= Image.open('versicolor.png')
    virginica = Image.open('virginica.png')

    #sidebar
    st.sidebar.title("Features")
    sl = st.sidebar.slider(label="Sepal Length (cm)",value=5.2,min_value=0.0, max_value=8.0, step=0.1)
    sw = st.sidebar.slider(label="Sepal Width (cm)",value=3.2,min_value=0.0, max_value=8.0, step=0.1)
    pl = st.sidebar.slider(label="Petal Length (cm)",value=4.2,min_value=0.0, max_value=8.0, step=0.1)
    pw = st.sidebar.slider(label="Petal Width (cm)",value=1.2,min_value=0.0, max_value=8.0, step=0.1)

    #action button
    if st.button("Click Here to Classify"):
        dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        input_variables = np.array(dfvalues[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
        prediction = knn_clf.predict(input_variables)
        if prediction == 1:
            st.image(setosa)
        elif prediction == 2:
            st.image(versicolor)
        elif prediction == 3:
            st.image(virginica)
    
if __name__=='__main__':
    main()