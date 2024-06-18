import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv('euro2024_players.csv')

# Set the page title
st.set_page_config(page_title="Euro 2024 Players")

# Display the title
st.title("Euro 2024 Players")

# Display the data as a table
st.dataframe(df)

# Allow users to filter the data
st.sidebar.title("Filter Players")
position = st.sidebar.multiselect("Select Position", df['Position'].unique())
country = st.sidebar.multiselect("Select Country", df['Country'].unique())

# Filter the data based on user selections
filtered_df = df
if position:
    filtered_df = filtered_df[filtered_df['Position'].isin(position)]
if country:
    filtered_df = filtered_df[filtered_df['Country'].isin(country)]

# Display the filtered data
st.subheader("Filtered Players")
st.dataframe(filtered_df)