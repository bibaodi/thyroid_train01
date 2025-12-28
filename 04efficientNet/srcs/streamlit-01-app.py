import streamlit as st

# Set the page title
st.title('My First Streamlit App')

# Display some text
st.write('This is a simple Streamlit application.')


st.write('This is a normal text')
number = 42
st.write(f'The number is {number}')


import pandas as pd
import numpy as np

data = {'Category': ['A', 'B', 'C'], 'Value': [10, 20, 15]}
df = pd.DataFrame(data)

st.bar_chart(df.set_index('Category'))

if st.button('Click me'):
    st.write('You clicked the button!')