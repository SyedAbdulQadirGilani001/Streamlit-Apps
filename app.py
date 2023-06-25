# import libraries
import streamlit as st # for the web app
import yfinance as yf # for the stock data 
import pandas as pd # for data manipulation
import numpy as np # for data manipulation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for advanced data visualization
import plotly.graph_objects as go # for more advanced data visualization
import plotly.express as px # for 3D data visualization
import datetime # for date manipulation
from datetime import date, timedelta # for date manipulation
from statsmodels.tsa.seasonal import seasonal_decompose # for time series analysis
import statsmodels.api as sm # for time series analysis
from statsmodels.tsa.stattools import adfuller # for time series analysis
# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market of the next 30 days')
# add a image from online resourcepip
st.image('https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg')
# take input from the user of app about the start and end date
# sidebar
st.sidebar.header('Select the parameters from below')
start_date=st.sidebar.date_input('Start Date',date(2022,5,30))
end_date=st.sidebar.date_input('End Date',date(2023,5,30))
# add ticker label list
ticker_list=['AAPL','MSFT','GOOGL','META','TSLA','NVDA','ADBE','PYPL','INTC','CMCSA','NFLX','PEP'] # list of tickers of companies like Apple, Microsoft, Google, Facebook, Tesla, Nvidia, Adobe, Paypal, Intel, Comcast, Netflix, Pepsi
ticker=st.sidebar.selectbox('Select the company',ticker_list) # select the company from the list
# fetch the data from user input using yfinance library
data=yf.download(ticker,start=start_date,end=end_date)
# add Data as a column to the dataframe
data.insert(0,'Date',data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data From',start_date,'to',end_date)
st.write(data.head())
# plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
fig=px.line(data,x='Date',y=data.columns,title='Closing Price of the stock',width=1000,height=600)
st.plotly_chart(fig)
# add a select box to select column from data
column=st.selectbox('Select the column to be used for forecasting',data.columns[1:])
# subsetting the data
data=data[['Date',column]]
st.write('Selected data')
st.write(data.head())
# ADF test check for stationarity
st.header('Is the data stationary?')
st.write('**Note:** If p-value is less than 0.05 then the data is stationary')
# ADF test
st.write('p-value of the data is',adfuller(data[column])[1] < 0.05)
# lets decompose the data
st.header('Decomposition of the data')
# decomposition
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write('Decomposed data',decomposition.plot())
# make the same plot using plotly
st.write('## Plotting the decomposed data using plotly')
# st.plotly_chart update traces
st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Trend of the data',width=1000,height=400,labels={'x':'Date','y':'Trend'}).update_traces(line_color='limegreen'))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.seasonal,title='Seasonality of the data',width=1000,height=400,labels={'x':'Date','y':'Seasonality'}).update_traces(line_color='darkorange'))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.resid,title='Residual of the data',width=1000,height=400,labels={'x':'Date','y':'Residual'}).update_traces(line_color='maroon'))
# lets run the ARIMA model
# lets ask the user for the parameters of the model for p,d,q
p=st.slider('Select the value of p',0,5,1) # slider for selecting the value of p 
d=st.slider('Select the value of d',0,5,1) # slider for selecting the value of d
q=st.slider('Select the value of q',0,5,1) # slider for selecting the value of q 1 is the default value 0 is the minimum value and 5 is the maximum value
seasonal_order=st.number_input('Enter the value of seasonal p',0,24,12)
model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
results=model.fit()
st.write('Summary of the model')
st.write(results.summary())
st.write('-----')
st.write('<p style="color:lime; font-size:50px;" font-weight:bold;>Forecasting the future values</p>',unsafe_allow_html=True)
# predict the future values forcecast
forecast_period=st.number_input('## Enter the number of days to forecast',1,365,30)
predictions=results.get_prediction(start=len(data),end=len(data)+forecast_period-1)
predictions=predictions.predicted_mean
predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions.head(forecast_period))
predictions.insert(0,'Date',predictions.index,True)
predictions.reset_index(drop=True,inplace=True)
st.write(f'## Predictions',predictions.head())
st.write('## Actual data',data.head())
# lets plot the data
fig=go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data[column],name='Actual Data',line=dict(color='limegreen',width=2)))
fig.add_trace(go.Scatter(x=predictions['Date'],y=predictions['predicted_mean'],name='Predicted Data',line=dict(color='darkorange',width=2)))
fig.update_layout(title='Actual vs Predicted Data',xaxis_title='Date',yaxis_title='Closing Price of the stock')
st.plotly_chart(fig)
# add buttons to show and hide separate plots
show_plots=False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data['Date'],y=data[column],title='Actual Data',width=1000,height=400,labels={'x':'Date','y':'Actual Data'}).update_traces(line_color='limegreen'))
        st.write(px.line(x=predictions['Date'],y=predictions['predicted_mean'],title='Predicted Data',width=1000,height=400,labels={'x':'Date','y':'Predicted Data'}).update_traces(line_color='darkorange'))
        show_plots=True
    else:
        show_plots=False
hide_plots=False
if st.button('Hide Separate Plots'):
    if not hide_plots:
        st.write(px.line(x=data['Date'],y=data[column],title='Actual Data',width=1000,height=400,labels={'x':'Date','y':'Actual Data'}).update_traces(line_color='limegreen'))
        st.write(px.line(x=predictions['Date'],y=predictions['predicted_mean'],title='Predicted Data',width=1000,height=400,labels={'x':'Date','y':'Predicted Data'}).update_traces(line_color='darkorange'))
        hide_plots=True
    else:
        hide_plots=False
st.write('-----')
st.write('## Thankyou for using the app')

st.write('## Made with ❤️')
st.write('-----')
st.write('#### Author: Syed Abdul Qadir Gilani')
st.write('#### Connect with me on Social Media')
linkedin_url='https://img.icons8.com/color/48/000000/linkedin.png'
github_url='https://img.icons8.com/fluent/48/000000/github.png'
twitter_url='https://img.icons8.com/color/48/000000/twitter.png'


linkedin_redirect_url='https://www.linkedin.com/in/syedabdulqadir/'
github_redirect_url='https://github.com/SyedAbdulQadirGilani001'
twitter_redirect_url='https://twitter.com/@QadirSyed42820'

st.markdown(f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}"></a>',unsafe_allow_html=True)
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}"></a>',unsafe_allow_html=True)
st.markdown(f'<a href="{twitter_redirect_url}"><img src="{twitter_url}"></a>',unsafe_allow_html=True)
st.write('-----')