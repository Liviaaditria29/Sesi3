import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('euro2024_players.csv')

# Sidebar for user input
st.sidebar.title("Euro 2024 Player Prediction")
st.sidebar.write("Configure the model parameters:")

# Get user inputs from the sidebar
age = st.sidebar.slider("Age", 18, 40, 25)
position = st.sidebar.selectbox("Position", df['Position'].unique())
club = st.sidebar.text_input("Club", "")
country = st.sidebar.selectbox("Country", df['Country'].unique())

# Filter the data based on user inputs
filtered_df = df[(df['Age'] == age) & (df['Position'] == position) & (df['Club'] == club) & (df['Country'] == country)]

# Display the filtered data
st.title("Euro 2024 Player Prediction")
st.write("Based on your inputs, the potential players are:")
st.dataframe(filtered_df)

# Perform the prediction
X = df.drop(['Name'], axis=1)
y = df['Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.write(f"The model accuracy is: {accuracy:.2f}")
