import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, genextreme
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def criarDataSet():
    # url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    # df = pd.read_csv(url, index_col=0)
    # data = df.query('location == "India" | location == "Brazil" | location == "Iran" | location == "South Africa" | location== "Egypt"')
    df = pd.read_csv('nd_covid_Grupo_Orange.csv', index_col=0)
    # data.to_csv('nd_covid_Grupo_Orange.csv', index=False, header=True)
    return df

def plotVisualizacao(dadoPais, titulo, filename, cat):
    if "Countries" in titulo:
        plt.plot(dadoPais[0], color='blue', label='Brazil')
        plt.title('Distribuition of ' + titulo)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel(cat)
        plt.savefig(filename)
        plt.show()


def obterVisualizacao(dado):
    a=[]
    ################################################################## DOS PAÍSES

    dadoBrazil = dado.query('location == "Brazil"').replace(np.nan, 0)
    NDCBrazil = dadoBrazil['new_cases'].values


    a.append(pd.DataFrame(NDCBrazil, columns=['Brazil']))
    plotVisualizacao(a, 'Countries New Daily Cases', "Coutries New Daily Cases.png", "New Cases")

def main():
    dado = criarDataSet()

    obterVisualizacao(dado)


main()
