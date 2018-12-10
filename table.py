from flask import Flask
#from flask import render_template
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import time
import re
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, linear_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import warnings
import holoviews as hv #This has to be downloaded via Anaconda ,as the import might not work
from bokeh.plotting import figure, output_file, show

hv.extension('bokeh')

# Reading the csv file
df = pd.read_csv('CA-CIB DB Extract.csv' , index_col=4, encoding = 'unicode_escape')

app = Flask(__name__)

#@app.route('/plot')
@app.route('/')
def build_plot():
    new_df= df
    img = io.BytesIO()
    emtn_df = new_df[new_df['Payoff'].str.contains("EMTN")]
    emtn_df.drop((emtn_df[emtn_df['Product Creator'].str.contains('Nino') == True]).index, inplace=True)
    dupes1 = emtn_df[(emtn_df.duplicated('Product IDs'))]
    emtn_df.drop(emtn_df[(emtn_df.duplicated('Product IDs'))].index, inplace=True)
    traded_df = (emtn_df.loc[emtn_df['Traded'] == "Traded"])
    dupes = emtn_df[(emtn_df.duplicated(subset=['Product Name', 'Underlying(s)']))]
    dupes.drop((dupes.loc[dupes['Traded'] == "Traded"]).index)
    emtn_df = pd.concat([emtn_df, dupes]).drop_duplicates(keep=False)
    emtn_df['SRI'].isnull().sum(axis=0)
    # missing= [emtn_df['SRI'].isnull().sum(axis=0)]
    mis_sri = emtn_df[emtn_df['SRI'].isnull()]
    emtn_df = pd.concat([emtn_df, mis_sri]).drop_duplicates(keep=False)
    emtn_df['RIY @RHP (%)'].mean(axis=0)
    # Calculating the Median for RIY@ RHP
    med = emtn_df['RIY @RHP (%)'].median(axis=0)
    # Filling the Median into NaN values of RIY
    emtn_df['RIY @RHP (%)'] = emtn_df['RIY @RHP (%)'].fillna(value=med)
    emtn_df['Moderate Return'].mean(axis=0)
    # Calculating the Median for Moderate Return
    med1 = emtn_df['Moderate Return'].median(axis=0)
    # Filling the Median into NaN values of Moderate Return
    emtn_df['Moderate Return'] = emtn_df['Moderate Return'].fillna(value=med1)
    emtn_df['Moderate Price (%)'].mean(axis=0)
    # Calculating the Median for Moderate Price (%)
    med2 = emtn_df['Moderate Price (%)'].median(axis=0)
    # Filling the Median into NaN values of Moderate Price (%)
    emtn_df['Moderate Price (%)'] = emtn_df['Moderate Price (%)'].fillna(value=med2)
    mis_sri['RIY @RHP (%)'] = mis_sri['RIY @RHP (%)'].fillna(value=med)
    mis_sri['Moderate Return'] = mis_sri['Moderate Return'].fillna(value=med1)
    mis_sri['Moderate Price (%)'] = mis_sri['Moderate Price (%)'].fillna(value=med2)
    # Applying the Model on mis_sri
    X1 = mis_sri[['FX Rate', 'Moderate Price (%)', 'Moderate Return', 'RIY @RHP (%)']]
    # mis_sri['SRI prediction'] = all_lm.predict(X1)
    # mis_sri['SRI prediction'] = mis_sri['SRI prediction'].astype(int)
    # mis_sri['SRI'] = mis_sri['SRI prediction']
    # emtn_df['SRI prediction'] = all_lm.predict(X)
    # emtn_df['SRI prediction'] = emtn_df['SRI prediction'].astype(int)
    new = pd.concat([emtn_df, mis_sri], verify_integrity=True)
    payoffmean = new.groupby(['Payoff']).SRI.mean()
    payoffmean = payoffmean.sort_values(ascending=False)
    sns.set(rc={'figure.figsize': (15, 15)})
    # sns.countplot(x=(new.groupby(['Payoff']).SRI.mean()), data=new )
    payoffmean.plot(kind='bar', figsize=(25, 10), fontsize=10)

    #plt.plot(df['Payoff'])
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

@app.route('/plot')
def build_plot1():
    for i in range(0, len(df.axes[1])):
        col_nulls = df.iloc[:, i].isnull().sum()
        precent = (col_nulls / len(df.axes[0])) * 100
        print(df.columns[i], col_nulls, '%.2f' % precent, "%")
    img = io.BytesIO()

    # y = [1,2,3,4,5]
    # x = [0,2,1,3,4]
    # #plt.plot(x,y)
    # plt.plot(df['Payoff'])
    # plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

if __name__ == '__main__':
    app.debug = True
    app.run()