import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cullen_frey_giovanni_trab import cullenfrey
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, genextreme
from sklearn.preprocessing import MinMaxScaler
import statsfuncs
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import mfdfa
import waipy

####################################################################### Criação do Dataset

def criarDataSet():
    # url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    # df = pd.read_csv(url, index_col=0)
    # data = df.query('location == "India" | location == "Brazil" | location == "Iran" | location == "South Africa" | location== "Egypt"')
    df = pd.read_csv('nd_covid_Grupo_Orange.csv', index_col=0)
    # data.to_csv('nd_covid_Grupo_Orange.csv', index=False, header=True)
    return df


def criarDataSetRegional():
    # CSV obtido em: https://www.seade.gov.br/coronavirus/

    df = pd.read_csv('dados_covid_sp.csv', index_col=0)
    data = df.query('nome_munic == "São José dos Campos" | nome_munic == "São Paulo"')
    return data


####################################################################### Obtenção dos Mapas de Cullen and Frey

def plotCullenFrey(dado, dadoRegional):
    ################################################################## DOS PAÍSES

    # Gera os mapas de Cullen-Frey dos casos totais

    tipoDado = ""
    df = []
    skew = []
    curt = []
    country = []
    a = []

    df.append(dado.query('location == "Brazil" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0))
    df.append(dado.query('location == "India" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0))
    df.append(dado.query('location == "Iran" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0))
    df.append(dado.query('location == "South Africa" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0))
    df.append(dado.query('location == "Egypt" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0))

    for i in range(4):
        if i == 0:
            tipoDado = "total_cases"
        if i == 1:
            tipoDado = "total_deaths"
        if i == 2:
            tipoDado = "new_cases"
        if i == 3:
            tipoDado = "new_deaths"

        for j in range(5):
            skew.append(statsfuncs.skewness(df[j][tipoDado].tolist()))
            curt.append(statsfuncs.kurtosis(df[j][tipoDado].tolist()))
            c = df[j].iloc[0]
            c = c['location']
            country.append(c)
            a.append([country[j],skew[j], curt[j]])
        cullenfrey(a, 'Countries ' + tipoDado.replace("_", " "))
        a.clear()
        skew.clear()
        curt.clear()
        country.clear()
    df.clear()

################################################################## DAS CIDADES REGIONAIS

    # Separa os dados regionais em diferentes DataFrames
    df.append(dadoRegional.query('nome_munic == "São José dos Campos"').replace(np.nan, 0))
    df.append(dadoRegional.query('nome_munic == "São Paulo"').replace(np.nan, 0))

    DataType = ""

    for i in range(4):
        if i == 0:
            tipoDado = "casos"
            DataType = "total cases"
        if i == 1:
            tipoDado = "obitos"
            DataType = "total deaths"
        if i == 2:
            tipoDado = "casos_novos"
            DataType = "new cases"
        if i == 3:
            tipoDado = "obitos_novos"
            DataType = "new deaths"

        for j in range(2):
            skew.append(statsfuncs.skewness(df[j][tipoDado].tolist()))
            curt.append(statsfuncs.kurtosis(df[j][tipoDado].tolist()))
            c = df[j].iloc[0]
            country.append(c.name)
            a.append([country[j],skew[j], curt[j]])
        cullenfrey(a, 'Regions ' + DataType)
        a.clear()
        skew.clear()
        curt.clear()
        country.clear()
    df.clear()

####################################################################### Criação e Plotagem da Visualização dos Dados

def plotVisualizacao(dadoPais, titulo, filename, cat):
    if "Countries" in titulo:
        plt.plot(dadoPais[0], color='blue', label='Brazil')
        plt.plot(dadoPais[1], color='orange', label='India')
        plt.plot(dadoPais[2], color='lightgreen', label='South Africa')
        plt.plot(dadoPais[3], color='yellow', label='Egypt')
        plt.plot(dadoPais[4], color='red', label='Iran')
        plt.title('Distribuition of ' + titulo)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel(cat)
        plt.savefig('Visualizacao/Visualizacao ' + filename)
        plt.show()

    else:
        plt.plot(dadoPais[0], color='blue', label='São José dos Campos')
        plt.plot(dadoPais[1], color='orange', label='São Paulo')
        plt.title('Distribuition of ' + titulo)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel(cat)
        plt.savefig('Visualizacao/Visualizacao ' + filename)
        plt.show()


def obterVisualizacao(dado, dadoRegional):
    ################################################################## DOS PAÍSES
    a = []

    dadoBrazil = dado.query('location == "Brazil"').replace(np.nan, 0)
    NDCBrazil = dadoBrazil['new_cases'].values
    NDTBrazil = dadoBrazil['new_tests'].values
    NDDBrazil = dadoBrazil['new_deaths'].values
    TCBrazil = dadoBrazil['total_cases'].values
    TDBrazil = dadoBrazil['total_deaths'].values
    TTBrazil = dadoBrazil['total_tests'].values

    dadoIndia = dado.query('location == "India"').replace(np.nan, 0)
    NDCIndia = dadoIndia['new_cases'].values
    NDTIndia = dadoIndia['new_tests'].values
    NDDIndia = dadoIndia['new_deaths'].values
    TCIndia = dadoIndia['total_cases'].values
    TDIndia = dadoIndia['total_deaths'].values
    TTIndia = dadoIndia['total_tests'].values

    dadoIran = dado.query('location == "Iran"').replace(np.nan, 0)
    NDCIran = dadoIran['new_cases'].values
    NDTIran = dadoIran['new_tests'].values
    NDDIran = dadoIran['new_deaths'].values
    TCIran = dadoIran['total_cases'].values
    TDIran = dadoIran['total_deaths'].values
    TTIran = dadoIran['total_tests'].values

    dadoAf = dado.query('location == "South Africa"').replace(np.nan, 0)
    NDCAf = dadoAf['new_cases'].values
    NDTAf = dadoAf['new_tests'].values
    NDDAf = dadoAf['new_deaths'].values
    TCAf = dadoAf['total_cases'].values
    TDAf = dadoAf['total_deaths'].values
    TTAf = dadoAf['total_tests'].values

    dadoEgypt = dado.query('location == "Egypt"').replace(np.nan, 0)
    NDCEgypt = dadoEgypt['new_cases'].values
    NDTEgypt = dadoEgypt['new_tests'].values
    NDDEgypt = dadoEgypt['new_deaths'].values
    TCEgypt = dadoEgypt['total_cases'].values
    TDEgypt = dadoEgypt['total_deaths'].values
    TTEgypt = dadoEgypt['total_tests'].values

    a.append(pd.DataFrame(NDCBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(NDCIndia, columns=['India']))
    a.append(pd.DataFrame(NDCAf, columns=['South Africa']))
    a.append(pd.DataFrame(NDCEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(NDCIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries New Daily Cases', "Coutries New Daily Cases.png", "New Cases")
    a.clear()

    a.append(pd.DataFrame(NDTBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(NDTIndia, columns=['India']))
    a.append(pd.DataFrame(NDTAf, columns=['South Africa']))
    a.append(pd.DataFrame(NDTEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(NDTIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries New Daily Tests', "Coutries New Daily Tests.png", "New Test")
    a.clear()

    a.append(pd.DataFrame(NDDBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(NDDIndia, columns=['India']))
    a.append(pd.DataFrame(NDDAf, columns=['South Africa']))
    a.append(pd.DataFrame(NDDEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(NDDIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries New Daily Deaths', "Coutries New Daily Deaths.png", "New Deaths")
    a.clear()

    a.append(pd.DataFrame(TCBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(TCIndia, columns=['India']))
    a.append(pd.DataFrame(TCAf, columns=['South Africa']))
    a.append(pd.DataFrame(TCEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(TCIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries Total Cases', "Coutries Total Cases.png", "Total Cases")
    a.clear()

    a.append(pd.DataFrame(TDBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(TDIndia, columns=['India']))
    a.append(pd.DataFrame(TDAf, columns=['South Africa']))
    a.append(pd.DataFrame(TDEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(TDIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries Total Deaths', "Coutries Total Deaths.png", "Total Deaths")
    a.clear()

    a.append(pd.DataFrame(TTBrazil, columns=['Brazil']))
    a.append(pd.DataFrame(TTIndia, columns=['India']))
    a.append(pd.DataFrame(TTAf, columns=['South Africa']))
    a.append(pd.DataFrame(TTEgypt, columns=['Egypt']))
    a.append(pd.DataFrame(TTIran, columns=['Iran']))
    plotVisualizacao(a, 'Countries Total Tests', "Coutries Total Tests.png", "Total Tests")
    a.clear()

    ################################################################## DAS CIDADES REGIONAIS

    dadoSjc = dadoRegional.query('nome_munic == "São José dos Campos"').replace(np.nan, 0)
    NDCSjc = dadoSjc['casos_novos'].values
    NDDSjc = dadoSjc['obitos_novos'].values
    TCSjc = dadoSjc['casos'].values
    TDSjc = dadoSjc['obitos'].values

    dadoSp = dadoRegional.query('nome_munic == "São Paulo"').replace(np.nan, 0)
    NDCSp = dadoSp['casos_novos'].values
    NDDSp = dadoSp['obitos_novos'].values
    TCSp = dadoSp['casos'].values
    TDSp = dadoSp['obitos'].values

    a.append(pd.DataFrame(NDCSjc, columns=['São José dos Campos']))
    a.append(pd.DataFrame(NDCSp, columns=['São Paulo']))
    plotVisualizacao(a, 'Regions New Daily Cases', "Regions New Daily Cases.png", "New Cases")
    a.clear()

    a.append(pd.DataFrame(TDSjc, columns=['São José dos Campos']))
    a.append(pd.DataFrame(TDSp, columns=['São Paulo']))
    plotVisualizacao(a, 'Regions New Daily Deaths', "Regions New Daily Deaths.png", "New Daily Deaths")
    a.clear()

    a.append(pd.DataFrame(TCSjc, columns=['São José dos Campos']))
    a.append(pd.DataFrame(TCSp, columns=['São Paulo']))
    plotVisualizacao(a, 'Regions Total Cases', "Regions Total Cases.png", "Total Cases")
    a.clear()

    a.append(pd.DataFrame(TDSjc, columns=['São José dos Campos']))
    a.append(pd.DataFrame(TDSp, columns=['São Paulo']))
    plotVisualizacao(a, 'Regions Total Deaths', "Regions Total Deaths.png", "Total Deaths")
    a.clear()


####################################################################### Criação e Plotagem dos Histogramas

def plotHistograma(dadoPais, titulo, filename, cat):
    plt.hist(dadoPais, bins=80, ec="k", alpha=0.6, color="royalblue")
    plt.title('Distribuição ' + titulo)
    plt.xlabel('Days')
    plt.ylabel(cat)
    plt.savefig('Histograma/Histograma ' + filename)
    plt.show()


def obterHistograma(dado, dadoRegional):
    ################################################################## DOS PAÍSES

    dadoBrazil = dado.query('location == "Brazil"').replace(np.nan, 0)
    NDCBrazil = dadoBrazil['new_cases'].values
    NDTBrazil = dadoBrazil['new_tests'].values
    NDDBrazil = dadoBrazil['new_deaths'].values
    TCBrazil = dadoBrazil['total_cases'].values
    TDBrazil = dadoBrazil['total_deaths'].values
    TTBrazil = dadoBrazil['total_tests'].values

    dadoIndia = dado.query('location == "India"').replace(np.nan, 0)
    NDCIndia = dadoIndia['new_cases'].values
    NDTIndia = dadoIndia['new_tests'].values
    NDDIndia = dadoIndia['new_deaths'].values
    TCIndia = dadoIndia['total_cases'].values
    TDIndia = dadoIndia['total_deaths'].values
    TTIndia = dadoIndia['total_tests'].values

    dadoIran = dado.query('location == "Iran"').replace(np.nan, 0)
    NDCIran = dadoIran['new_cases'].values
    NDTIran = dadoIran['new_tests'].values
    NDDIran = dadoIran['new_deaths'].values
    TCIran = dadoIran['total_cases'].values
    TDIran = dadoIran['total_deaths'].values
    TTIran = dadoIran['total_tests'].values

    dadoAf = dado.query('location == "South Africa"').replace(np.nan, 0)
    NDCAf = dadoAf['new_cases'].values
    NDTAf = dadoAf['new_tests'].values
    NDDAf = dadoAf['new_deaths'].values
    TCAf = dadoAf['total_cases'].values
    TDAf = dadoAf['total_deaths'].values
    TTAf = dadoAf['total_tests'].values

    dadoEgypt = dado.query('location == "Egypt"').replace(np.nan, 0)
    NDCEgypt = dadoEgypt['new_cases'].values
    NDTEgypt = dadoEgypt['new_tests'].values
    NDDEgypt = dadoEgypt['new_deaths'].values
    TCEgypt = dadoEgypt['total_cases'].values
    TDEgypt = dadoEgypt['total_deaths'].values
    TTEgypt = dadoEgypt['total_tests'].values

    plotHistograma(NDCBrazil, 'Brazil New Daily Cases', "Brazil New Daily Cases.png", "New Daily Cases")
    plotHistograma(NDCIndia, 'India New Daily Cases', "India New Daily Cases.png", "New Daily Cases")
    plotHistograma(NDCIran, 'Iran New Daily Cases', "Iran New Daily Cases.png", "New Daily Cases")
    plotHistograma(NDCAf, 'South Africa New Daily Cases', "South Africa New Daily Cases.png", "New Daily Cases")
    plotHistograma(NDCEgypt, 'Egypt New Daily Cases', "Egypt New Daily Cases.png", "New Daily Cases")

    plotHistograma(NDTBrazil, 'Brazil New Daily Tests', "Brazil New Daily Tests.png", "New Daily Tests")
    plotHistograma(NDTIndia, 'India New Daily Tests', "India New Daily Tests.png", "New Daily Tests")
    plotHistograma(NDTIran, 'Iran New Daily Tests', "Iran New Daily Tests.png", "New Daily Tests")
    plotHistograma(NDTAf, 'South Africa New Daily Tests', "South Africa New Daily Tests.png", "New Daily Tests")
    plotHistograma(NDTEgypt, 'Egypt New Daily Tests', "Egypt New Daily Tests.png", "New Daily Tests")

    plotHistograma(NDDBrazil, 'Brazil New Daily Deaths', "Brazil New Daily Deaths.png", "New Daily Deaths")
    plotHistograma(NDDIndia, 'India New Daily Deaths', "India New Daily Deaths.png", "New Daily Deaths")
    plotHistograma(NDDIran, 'Iran New Daily Deaths', "Iran New Daily Deaths.png", "New Daily Deaths")
    plotHistograma(NDDAf, 'South Africa New Daily Deaths', "South Africa New Daily Deaths.png", "New Daily Deaths")
    plotHistograma(NDDEgypt, 'Egypt New Daily Deaths', "Egypt New Daily Deaths.png", "New Daily Deaths")

    plotHistograma(TCBrazil, 'Brazil Total Cases', "Brazil Total Cases.png", "Total Cases")
    plotHistograma(TCIndia, 'India Total Cases', "India TotalCases.png", "Total Cases")
    plotHistograma(TCIran, 'Iran Total Cases', "Iran Total Cases.png", "Total Cases")
    plotHistograma(TCAf, 'South Africa Total Cases', "South Africa Total Cases.png", "Total Cases")
    plotHistograma(TCEgypt, 'Egypt Total Cases', "Egypt Total Cases.png", "Total Cases")

    plotHistograma(TDBrazil, 'Brazil Total Deaths', "Brazil Total Deaths.png", "Total Deaths")
    plotHistograma(TDIndia, 'India Total Deaths', "India Total Deaths.png", "Total Deaths")
    plotHistograma(TDIran, 'Iran Total Deaths', "Iran Total Deaths.png", "Total Deaths")
    plotHistograma(TDAf, 'South Africa Total Deaths', "South Africa Total Deaths.png", "Total Deaths")
    plotHistograma(TDEgypt, 'Egypt Total Deaths', "Egypt Total Deaths.png", "Total Deaths")

    plotHistograma(TTBrazil, 'Brazil Total Tests', "Brazil Total Tests.png", "Total Tests")
    plotHistograma(TTIndia, 'India Total Tests', "India Total Tests.png", "Total Tests")
    plotHistograma(TTIran, 'Iran Total Tests', "Iran Total Tests.png", "Total Tests")
    plotHistograma(TTAf, 'South Africa Total Tests', "South Africa Total Tests.png", "Total Tests")
    plotHistograma(TTEgypt, 'Egypt Total Tests', "Egypt Total Tests.png", "Total Tests")

    ################################################################## DAS CIDADES REGIONAIS

    dadoSjc = dadoRegional.query('nome_munic == "São José dos Campos"').replace(np.nan, 0)
    NDCSjc = dadoSjc['casos_novos'].values
    NDDSjc = dadoSjc['obitos_novos'].values
    TCSjc = dadoSjc['casos'].values
    TDSjc = dadoSjc['obitos'].values

    dadoSp = dadoRegional.query('nome_munic == "São Paulo"').replace(np.nan, 0)
    NDCSp = dadoSp['casos_novos'].values
    NDDSp = dadoSp['obitos_novos'].values
    TCSp = dadoSp['casos'].values
    TDSp = dadoSp['obitos'].values

    plotHistograma(NDCSjc, 'São José dos Campos New Daily Cases', "São José dos Campos New Daily Cases.png", "New Daily Cases")
    plotHistograma(NDCSp, 'São Paulo New Daily Cases', "São Paulo New Daily Cases.png", "New Daily Cases")

    plotHistograma(NDDSjc, 'São José dos Campos New Daily Deaths', "São José dos Campos New Daily Deaths.png", "New Daily Deaths")
    plotHistograma(NDDSp, 'São Paulo New Daily Deaths', "São Paulo New Daily Deaths.png", "New Daily Deaths")

    plotHistograma(TCSjc, 'São José dos Campos Total Cases', "São José dos Campos Total Cases.png", "Total Cases")
    plotHistograma(TCSp, 'São Paulo Total Cases', "São Paulo Total Cases.png", "Total Cases")

    plotHistograma(TDSjc, 'São José dos Campos Total Deaths', "São José dos Campos Total Deaths.png", "Total Deaths")
    plotHistograma(TDSp, 'São Paulo Total Deaths', "São Paulo Total Deaths.png", "Total Deaths")


####################################################################### Criação e Plotagem dos PDF      

def plotPDF(dadoPais, titulo, filename, country):
    n, bins, patches = plt.hist(dadoPais, 60, density=1, facecolor='mediumseagreen', alpha=0.75, label="Normalized data")
    x = range(len(dadoPais))
    ymin = min(dadoPais)
    ymax = max(dadoPais)
    n = len(dadoPais)
    ypoints = [(ymin + (i / n) * (ymax - ymin)) for i in range(0, n + 1)]

    # fit da GEV
    mu, sigma = norm.fit(dadoPais)
    rv_nrm = norm(loc=mu, scale=sigma)
    gev_fit = genextreme.fit(dadoPais)  # estimando GEV
    c, loc, scale = gev_fit
    mean, var, skew, kurt = genextreme.stats(c, moments='mvsk')
    rv_gev = genextreme(c, loc=loc, scale=scale)
    gev_pdf = rv_gev.pdf(ypoints)  # criando dados a partir do GEV estimado para plotar

    plt.title("PDF with data from " + titulo + "\nmu={0:3.5}, sigma={1:3.5}".format(mu, sigma))

    plt.plot(np.arange(min(bins), max(bins) + 1, (max(bins) - min(bins)) / len(dadoPais)), gev_pdf, 'orange', lw=5, alpha=0.6,
             label='genextreme pdf')
    n, bins, patches = plt.hist(dadoPais, 60, density=1, facecolor='mediumseagreen', alpha=0.75, label="Normalized data")
    plt.ylabel("Probability density")
    plt.xlabel("Value")
    plt.legend()
    plt.savefig("PDF/PDF" + filename)
    plt.show()


def obterPDF(dado, dadoRegional):
    ################################################################## DOS PAÍSES

    dadoBrazil = dado.query('location == "Brazil" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCBrazil = dadoBrazil['new_cases'].values
    NDTBrazil = dadoBrazil['new_tests'].values
    NDDBrazil = dadoBrazil['new_deaths'].values
    TCBrazil = dadoBrazil['total_cases'].values
    TDBrazil = dadoBrazil['total_deaths'].values
    TTBrazil = dadoBrazil['total_tests'].values

    dadoIndia = dado.query('location == "India" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCIndia = dadoIndia['new_cases'].values
    NDTIndia = dadoIndia['new_tests'].values
    NDDIndia = dadoIndia['new_deaths'].values
    TCIndia = dadoIndia['total_cases'].values
    TDIndia = dadoIndia['total_deaths'].values
    TTIndia = dadoIndia['total_tests'].values

    dadoIran = dado.query('location == "Iran" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCIran = dadoIran['new_cases'].values
    NDTIran = dadoIran['new_tests'].values
    NDDIran = dadoIran['new_deaths'].values
    TCIran = dadoIran['total_cases'].values
    TDIran = dadoIran['total_deaths'].values
    TTIran = dadoIran['total_tests'].values

    dadoAf = dado.query('location == "South Africa" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCAf = dadoAf['new_cases'].values
    NDTAf = dadoAf['new_tests'].values
    NDDAf = dadoAf['new_deaths'].values
    TCAf = dadoAf['total_cases'].values
    TDAf = dadoAf['total_deaths'].values
    TTAf = dadoAf['total_tests'].values

    dadoEgypt = dado.query('location == "Egypt" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCEgypt = dadoEgypt['new_cases'].values
    NDTEgypt = dadoEgypt['new_tests'].values
    NDDEgypt = dadoEgypt['new_deaths'].values
    TCEgypt = dadoEgypt['total_cases'].values
    TDEgypt = dadoEgypt['total_deaths'].values
    TTEgypt = dadoEgypt['total_tests'].values


    plotPDF(NDCBrazil, 'Brazil New Daily Cases', "Brazil New Daily Cases.png", "Brazil")
    plotPDF(NDCIndia, 'India New Daily Cases', "India New Daily Cases.png", "India")
    plotPDF(NDCIran, 'Iran New Daily Cases', "Iran New Daily Cases.png", "Iran")
    plotPDF(NDCAf, 'South Africa New Daily Cases', "South Africa New Daily Cases.png", "South Africa")
    plotPDF(NDCEgypt, 'Egypt New Daily Cases', "Egypt New Daily Cases.png", "Egypt")

    #plotPDF(NDTBrazil, 'Brazil New Daily Tests', "Brazil New Daily Tests.png", "Brazil")
    plotPDF(NDTIndia, 'India New Daily Tests', "India New Daily Tests.png", "India")
    plotPDF(NDTIran, 'Iran New Daily Tests', "Iran New Daily Tests.png", "Iran")
    plotPDF(NDTAf, 'South Africa New Daily Tests', "South Africa New Daily Tests.png", "South Africa")
    #plotPDF(NDTEgypt, 'Egypt New Daily Tests', "Egypt New Daily Tests.png", "Egypt")
    
    #plotPDF(NDDBrazil, 'Brazil New Daily Deaths', "Brazil New Daily Deaths.png", "Brazil")
    plotPDF(NDDIndia, 'India New Daily Deaths', "India New Daily Deaths.png", "India")
    plotPDF(NDDIran, 'Iran New Daily Deaths', "Iran New Daily Deaths.png", "Iran")
    #plotPDF(NDDAf, 'South Africa New Daily Deaths', "South Africa New Daily Deaths.png", "South Africa")
    #plotPDF(NDDEgypt, 'Egypt New Daily Deaths', "Egypt New Daily Deaths.png", "Egypt")

    plotPDF(TCBrazil, 'Brazil Total Cases', "Brazil Total Cases.png", "Brazil")
    plotPDF(TCIndia, 'India Total Cases', "India TotalCases.png", "India")
    plotPDF(TCIran, 'Iran Total Cases', "Iran Total Cases.png", "Iran")
    plotPDF(TCAf, 'South Africa Total Cases', "South Africa Total Cases.png", "South Africa")
    plotPDF(TCEgypt, 'Egypt Total Cases', "Egypt Total Cases.png", "Egypt")

    plotPDF(TDBrazil, 'Brazil Total Deaths', "Brazil Total Deaths.png", "Brazil")
    plotPDF(TDIndia, 'India Total Deaths', "India Total Deaths.png", "India")
    plotPDF(TDIran, 'Iran Total Deaths', "Iran Total Deaths.png", "Iran")
    plotPDF(TDAf, 'South Africa Total Deaths', "South Africa Total Deaths.png", "South Africa")
    plotPDF(TDEgypt, 'Egypt Total Deaths', "Egypt Total Deaths.png", "Egypt")

    plotPDF(TTBrazil, 'Brazil Total Tests', "Brazil Total Tests.png", "Brazil")
    plotPDF(TTIndia, 'India Total Tests', "India Total Tests.png", "India")
    plotPDF(TTIran, 'Iran Total Tests', "Iran Total Tests.png", "Iran")
    plotPDF(TTAf, 'South Africa Total Tests', "South Africa Total Tests.png", "South Africa")
    #plotPDF(TTEgypt, 'Egypt Total Tests', "Egypt Total Tests.png", "Egypt")

    ################################################################## DAS CIDADES REGIONAIS

    dadoSjc = dadoRegional.query('nome_munic == "São José dos Campos"').replace(np.nan, 0)
    NDCSjc = dadoSjc['casos_novos'].values
    NDDSjc = dadoSjc['obitos_novos'].values
    TCSjc = dadoSjc['casos'].values
    TDSjc = dadoSjc['obitos'].values

    dadoSp = dadoRegional.query('nome_munic == "São Paulo"').replace(np.nan, 0)
    NDCSp = dadoSp['casos_novos'].values
    NDDSp = dadoSp['obitos_novos'].values
    TCSp = dadoSp['casos'].values
    TDSp = dadoSp['obitos'].values

    #plotPDF(NDCSjc, 'São José dos Campos New Daily Cases', "São José dos Campos New Daily Cases.png", "São José dos Campos")
    plotPDF(NDCSp, 'São Paulo New Daily Cases', "São Paulo New Daily Cases.png", "São Paulo")

    plotPDF(NDDSjc, 'São José dos Campos New Daily Deaths', "São José dos Campos New Daily Deaths.png", "São José dos Campos")
    plotPDF(NDDSp, 'São Paulo New Daily Deaths', "São Paulo New Daily Deaths.png", "São Paulo")

    plotPDF(TCSjc, 'São José dos Campos Total Cases', "São José dos Campos Total Cases.png", "São José dos Campos")
    plotPDF(TCSp, 'São Paulo Total Cases', "São Paulo Total Cases.png", "São Paulo")

    #plotPDF(TDSjc, 'São José dos Campos Total Deaths', "São José dos Campos Total Deaths.png", "São José dos Campos")
    plotPDF(TDSp, 'São Paulo Total Deaths', "São Paulo Total Deaths.png", "São Paulo")


####################################################################### Criação e Plotagem da Regressão linear
####################################################################### dos NDC e NDT parecidos

def plotarRegressaoLinear(rvsPais1, rvsPais2, titulo, legenda, cat):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rvsPais1Norm = scaler.fit_transform(pd.DataFrame(rvsPais1))
    rvsPais2Norm = scaler.fit_transform(pd.DataFrame(rvsPais2))

    correlacao = np.corrcoef(rvsPais1Norm, rvsPais2Norm)

    rvsPais1Norm = rvsPais1Norm.reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(rvsPais1Norm, rvsPais2Norm)

    plt.scatter(rvsPais1Norm, rvsPais2Norm)
    plt.plot(rvsPais1Norm, regressor.predict(rvsPais1Norm), color='red')
    plt.title("Regressão Linear " + titulo + " - " + legenda)
    plt.xlabel('Days')
    plt.ylabel(cat)
    plt.savefig("Regressao/Regressão Linear " + titulo + " - " + legenda + ".png")
    plt.show()


def obterRegressaoLinear(dado, dadoRegional):
    ################################################################## DOS PAÍSES

    dadoBrazil = dado.query('location == "Brazil" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCBrazil = dadoBrazil['new_cases'].values
    NDTBrazil = dadoBrazil['new_tests'].values

    dadoIndia = dado.query('location == "India" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCIndia = dadoIndia['new_cases'].values
    NDTIndia = dadoIndia['new_tests'].values

    dadoIran = dado.query('location == "Iran" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCIran = dadoIran['new_cases'].values
    NDTIran = dadoIran['new_tests'].values

    dadoAf = dado.query('location == "South Africa" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCAf = dadoAf['new_cases'].values
    NDTAf = dadoAf['new_tests'].values

    dadoEgypt = dado.query('location == "Egypt" & date >= "2020-03-10" & date <= "2020-05-28"').replace(np.nan, 0)
    NDCEgypt = dadoEgypt['new_cases'].values
    NDTEgypt = dadoEgypt['new_tests'].values

    # Regressão NDC entre Brazil e South Africa
    plotarRegressaoLinear(NDCBrazil, NDCAf, "Brazil e India", "NDC", "New Daily Cases")

    # Regressão NDC entre Iran e India
    plotarRegressaoLinear(NDCIran, NDCIndia, "Iran e India", "NDC", "New Daily Cases")

    # Regressão NDC entre Iran e Egypt
    plotarRegressaoLinear(NDCIran, NDCEgypt, "Iran e Egypt", "NDC", "New Daily Cases")

    # Regressão NDC entre India e Egypt
    plotarRegressaoLinear(NDCIndia, NDCEgypt, "India e Egypt", "NDC", "New Daily Cases")

    # Regressão NDT entre Brazil e Egypt
    plotarRegressaoLinear(NDTBrazil, NDTEgypt, "Brazil e Egypt", "NDT", "New Daily Tests")

    # Regressão NDT entre Iran e India
    plotarRegressaoLinear(NDTIran, NDTIndia, "Iran e India", "NDT", "New Daily Tests")

    ################################################################## DAS CIDADES REGIONAIS

    dadoSjc = dadoRegional.query('nome_munic == "São José dos Campos"').replace(np.nan, 0)
    NDCSjc = dadoSjc['casos_novos'].values

    dadoSp = dadoRegional.query('nome_munic == "São Paulo"').replace(np.nan, 0)
    NDCSp = dadoSp['casos_novos'].values

    # Regressão NDC entre São José dos Campos e São Paulo
    plotarRegressaoLinear(NDCSjc, NDCSp, "São José dos Campos e São Paulo", "NDC", "New Daily Cases")


####################################################################### Cálculo da curva G e S para previsão dos casos diários
####################################################################### de COVID por país

def calculateGandSCurvePais(dado, country):
    p = [[0.5, 0.45, 0.05], [0.7, 0.25, 0.05]]
    pind = 0
    vals = [[2, 4, 5], [4, 7, 10]]

    y = dado['new_cases'].values.tolist()
    date = dado['date'].values.tolist()

    # Quantidade de dias que serão usados
    meandays = 7

    Nmin = []
    Nmax = []
    Nguess = []
    Nk7 = []
    g = []
    deltank = []

    xticks = []
    for i in range(meandays, len(date)):
        if i % meandays == 0:
            xticks.append(date[i])

    for i in range(meandays, len(y)):
        Nk7.append((sum(y[i - meandays:i])) / meandays)
        if y[i] < Nk7[-1]:
            g.append(y[i] / Nk7[-1])
        else:
            g.append(Nk7[-1] / y[i])
        n = np.dot(p[pind], y[i])
        Nmin.append(g[-1] * np.dot(n, vals[0]))
        Nmax.append(g[-1] * np.dot(n, vals[1]))
        Nguess.append((Nmin[-1] + Nmax[-1]) / 2)
        if (y[i] != 0):
            deltank.append((Nk7[-1] - y[i]) / y[i])
        else:
            deltank.append(np.nan)

    # Calculating deltag to calculate s and plot
    deltag = [0]
    for i in range(1, len(g)):
        g0 = g[i - 1]
        if g0 < g[i]:
            deltag.append(g0 - g[i] - (1 - g[i]) ** 2)
        else:
            deltag.append(g0 - g[i] + (1 - g0) ** 2)

    deltag = np.array(deltag)
    deltank = np.array(deltank)
    s = (2 * deltag + deltank) / 3

    # Plotando as variáveis Nmin, Nmax, Nguess com os dados originais

    plt.title("Graph with the data and the mean of 7 days for each data")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nk7)), Nk7, label="{} days means".format(meandays))
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}meananddata.png".format(country + ", "))
    plt.show()

    plt.title("Original Data with predictions")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nguess)), Nguess, label="Predict")
    plt.xticks(np.arange(80, step=meandays), xticks, rotation=45)
    plt.plot(range(len(Nmin)), Nmin, label="Nmin")
    plt.plot(range(len(Nmax)), Nmax, label="Nmax")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}originaldata.png".format(country + ", "))
    plt.show()

    # Plotando os valores calculados para g

    g = np.array(g)
    plt.figure(figsize=(20, 10))
    meang = abs(sum(g) / len(g) - g)
    plt.title("Values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.errorbar(range(len(g)), g, yerr=meang, xerr=0, hold=True, ecolor='k',
                 fmt='none', label='data', elinewidth=0.5, capsize=1)
    plt.plot(range(len(g)), g, 'o-')
    plt.savefig("CurvaGeS/" + country + "/{}originalg.png".format(country + ", "))
    plt.show()

    # Plotando os valores calculados para s

    s = np.array(s)
    plt.figure(figsize=(20, 10))
    means = abs(sum(s) / len(s) - s)
    plt.title("Values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.errorbar(range(len(s)), s, yerr=meang, xerr=0, hold=True, ecolor='k', fmt='none', label='data', elinewidth=0.5,
                 capsize=1)
    plt.plot(range(len(s)), s, 'o-')
    plt.savefig("CurvaGeS/" + country + "/{}originals.png".format(country + ", "))
    plt.show()

    # Previsão a partir dos dados originais

    preddays = 20
    predictNmin = [Nmin[-1]]
    predictNmax = [Nmax[-1]]
    predictg = []
    predictNmed = y[-meandays - 1:]
    predictNk7 = []
    predictdeltank = []
    for i in range(meandays, preddays + meandays):
        predictNk7.append(sum(predictNmed[i - meandays:i]) / meandays)
        if predictNmed[i] < predictNk7[-1]:
            predictg.append(predictNmed[i] / predictNk7[-1])
        else:
            predictg.append(predictNk7[-1] / predictNmed[i])
        n = np.dot(p[pind], predictNmed[-1])
        predictNmin.append(predictg[-1] * np.dot(n, vals[0]))
        predictNmax.append(predictg[-1] * np.dot(n, vals[1]))
        predictNmed.append((predictNmin[-1] + predictNmax[-1]) / 2)
        predictdeltank.append((predictNk7[-1] - predictNmed[-1]) / predictNmed[-1])

    plt.title("Plot showing the prediction for the next {} days".format(preddays))
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nguess)), Nguess, label="Nmed", c="orange")
    plt.plot(range(len(y) - meandays - 1, len(y) + preddays - meandays),
             predictNmed[meandays:], c="orange", linestyle='--',
             label="Predict Nmed")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predictmeananddata.png".format(country + ", "))
    plt.show()

    plt.title("Predict values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.plot(range(len(g)), g, c="b", label="g from data")
    plt.plot(range(len(g) - 1, len(g) + preddays - 1), predictg, c="b", linestyle='--', label="Generated g")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predictg.png".format(country + ", "))
    plt.show()

    predictdeltag = [0]
    for i in range(1, len(predictg)):
        g0 = predictg[i - 1]
        if g0 < predictg[i]:
            predictdeltag.append(g0 - predictg[i] - (1 - predictg[i]) ** 2)
        else:
            predictdeltag.append(g0 - predictg[i] + (1 - g0) ** 2)

    predictdeltag = np.array(predictdeltag)
    predictdeltank = np.array(predictdeltank)
    predicts = (2 * predictdeltag + predictdeltank) / 3

    plt.title("Predict values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.plot(range(len(s)), s, c="b", label="g from data")
    plt.plot(range(len(s) - 1, len(s) + preddays - 1), predicts, c="b",
             linestyle='--', label="Generated s")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predicts.png".format(country + ", "))
    plt.show()


def calculateGandSCurveRegiao(dado, country):
    p = [[0.5, 0.45, 0.05], [0.7, 0.25, 0.05]]
    pind = 0
    vals = [[2, 4, 5], [4, 7, 10]]

    y = dado['casos_novos'].values.tolist()
    date = dado['datahora'].values.tolist()

    # Quantidade de dias que serão usados
    meandays = 7

    Nmin = []
    Nmax = []
    Nguess = []
    Nk7 = []
    g = []
    deltank = []

    xticks = []
    for i in range(meandays, len(date)):
        if i % meandays == 0:
            xticks.append(date[i])

    for i in range(meandays, len(y)):
        Nk7.append((sum(y[i - meandays:i])) / meandays)
        if y[i] < Nk7[-1]:
            g.append(y[i] / Nk7[-1])
        else:
            g.append(Nk7[-1] / y[i])
        n = np.dot(p[pind], y[i])
        Nmin.append(g[-1] * np.dot(n, vals[0]))
        Nmax.append(g[-1] * np.dot(n, vals[1]))
        Nguess.append((Nmin[-1] + Nmax[-1]) / 2)
        if (y[i] != 0):
            deltank.append((Nk7[-1] - y[i]) / y[i])
        else:
            deltank.append(np.nan)

    # Calculating deltag to calculate s and plot
    deltag = [0]
    for i in range(1, len(g)):
        g0 = g[i - 1]
        if g0 < g[i]:
            deltag.append(g0 - g[i] - (1 - g[i]) ** 2)
        else:
            deltag.append(g0 - g[i] + (1 - g0) ** 2)

    deltag = np.array(deltag)
    deltank = np.array(deltank)
    s = (2 * deltag + deltank) / 3

    # Plotando as variáveis Nmin, Nmax, Nguess com os dados originais

    plt.title("Graph with the data and the mean of 7 days for each data")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nk7)), Nk7, label="{} days means".format(meandays))
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}meananddata.png".format(country + ", "))
    plt.show()

    plt.title("Original Data with predictions")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nguess)), Nguess, label="Predict")
    plt.xticks(np.arange(80, step=meandays), xticks, rotation=45)
    plt.plot(range(len(Nmin)), Nmin, label="Nmin")
    plt.plot(range(len(Nmax)), Nmax, label="Nmax")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}originaldata.png".format(country + ", "))
    plt.show()

    # Plotando os valores calculados para g

    g = np.array(g)
    plt.figure(figsize=(20, 10))
    meang = abs(sum(g) / len(g) - g)
    plt.title("Values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.errorbar(range(len(g)), g, yerr=meang, xerr=0, hold=True, ecolor='k',
                 fmt='none', label='data', elinewidth=0.5, capsize=1)
    plt.plot(range(len(g)), g, 'o-')
    plt.savefig("CurvaGeS/" + country + "/{}originalg.png".format(country + ", "))
    plt.show()

    # Plotando os valores calculados para s

    s = np.array(s)
    plt.figure(figsize=(20, 10))
    means = abs(sum(s) / len(s) - s)
    plt.title("Values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.errorbar(range(len(s)), s, yerr=meang, xerr=0, hold=True, ecolor='k', fmt='none', label='data', elinewidth=0.5,
                 capsize=1)
    plt.plot(range(len(s)), s, 'o-')
    plt.savefig("CurvaGeS/" + country + "/{}originals.png".format(country + ", "))
    plt.show()

    # Previsão a partir dos dados originais

    preddays = 20
    predictNmin = [Nmin[-1]]
    predictNmax = [Nmax[-1]]
    predictg = []
    predictNmed = y[-meandays - 1:]
    predictNk7 = []
    predictdeltank = []
    for i in range(meandays, preddays + meandays):
        predictNk7.append(sum(predictNmed[i - meandays:i]) / meandays)
        if predictNmed[i] < predictNk7[-1]:
            predictg.append(predictNmed[i] / predictNk7[-1])
        else:
            predictg.append(predictNk7[-1] / predictNmed[i])
        n = np.dot(p[pind], predictNmed[-1])
        predictNmin.append(predictg[-1] * np.dot(n, vals[0]))
        predictNmax.append(predictg[-1] * np.dot(n, vals[1]))
        predictNmed.append((predictNmin[-1] + predictNmax[-1]) / 2)
        predictdeltank.append((predictNk7[-1] - predictNmed[-1]) / predictNmed[-1])

    plt.title("Plot showing the prediction for the next {} days".format(preddays))
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y) - meandays), y[meandays:], label="Dados")
    plt.plot(range(len(Nguess)), Nguess, label="Nmed", c="orange")
    plt.plot(range(len(y) - meandays - 1, len(y) + preddays - meandays),
             predictNmed[meandays:], c="orange", linestyle='--',
             label="Predict Nmed")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predictmeananddata.png".format(country + ", "))
    plt.show()

    plt.title("Predict values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.plot(range(len(g)), g, c="b", label="g from data")
    plt.plot(range(len(g) - 1, len(g) + preddays - 1), predictg, c="b", linestyle='--', label="Generated g")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predictg.png".format(country + ", "))
    plt.show()

    predictdeltag = [0]
    for i in range(1, len(predictg)):
        g0 = predictg[i - 1]
        if g0 < predictg[i]:
            predictdeltag.append(g0 - predictg[i] - (1 - predictg[i]) ** 2)
        else:
            predictdeltag.append(g0 - predictg[i] + (1 - g0) ** 2)

    predictdeltag = np.array(predictdeltag)
    predictdeltank = np.array(predictdeltank)
    predicts = (2 * predictdeltag + predictdeltank) / 3

    plt.title("Predict values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.plot(range(len(s)), s, c="b", label="g from data")
    plt.plot(range(len(s) - 1, len(s) + preddays - 1), predicts, c="b",
             linestyle='--', label="Generated s")
    plt.legend()
    plt.savefig("CurvaGeS/" + country + "/{}predicts.png".format(country + ", "))
    plt.show()


def obterGandSCurve(dado, dadoRegional):
    ################################################################## DOS PAÍSES
    '''
    dadoBrazil = dado.query('total_cases > 50 & location == "Brazil" & date <= "2020-05-20"')
    dadoIndia = dado.query('total_cases > 50 & location == "India" & date <= "2020-05-20"')
    dadoEgypt = dado.query('total_cases > 50 & location == "Egypt" & date <= "2020-05-20"')
    dadoAf = dado.query('total_cases > 50 & location == "South Africa" & date <= "2020-05-20"')
    dadoIran = dado.query('total_cases > 50 & location == "Iran" & date <= "2020-05-20"')

    calculateGandSCurvePais(dadoBrazil, "Brazil")
    calculateGandSCurvePais(dadoIndia, "India")
    calculateGandSCurvePais(dadoEgypt, "Egypt")
    calculateGandSCurvePais(dadoAf, "South Africa")
    calculateGandSCurvePais(dadoIran, "Iran")
    '''
    ################################################################## DAS CIDADES REGIONAIS

    dadoSjc = dadoRegional.query('casos > 50 & nome_munic == "São José dos Campos"').replace(np.nan, 0)
    dadoSp = dadoRegional.query('casos > 50 & nome_munic == "São Paulo"').replace(np.nan, 0)

    calculateGandSCurveRegiao(dadoSjc, "São José Dos Campos")
    calculateGandSCurveRegiao(dadoSp, "São Paulo")


def SOC(data, title, n_bins=50):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    # print("mean: ", mean, " var: ", var)
    """ Computa a Taxa Local de Flutuação para cada valor da ST """
    Gamma = []

    for i in range(0, n):
        Gamma.append((data[i] - mean) / var)
        # Gamma.append((data[i] - mean)/std)

        """ Computa P[Psi_i] """

    counts, bins = np.histogram(Gamma, n_bins)
    Prob_Gamma = []
    for i in range(0, n_bins):
        Prob_Gamma.append(counts[i] / n)  # plt.plot(Gamma)
    log_Prob = np.log10(Prob_Gamma)
    p = np.array(Prob_Gamma)
    p = p[np.nonzero(p)]
    c = counts[np.nonzero(counts)]
    log_p = np.log10(p)
    a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c))
    b = log_Prob[0]
    y = b * np.power(10, (a * counts))

    """ Plotagem """

    plt.clf()
    plt.scatter(np.log10(counts), y, marker=".", color="blue")
    plt.title('SOC Country: {}'.format(title), fontsize=16)
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)')
    plt.grid()
    plt.show()

def obterSOC():
    namefile = "daily-cases-covid-19.csv"
    l = pd.read_csv(namefile)
    codes = list(set(l["Entity"]))
    codes = codes[1:]
    l = l.set_index("Entity")
    values = []
    countries = ["Brazil", "India", "Iran", "South Africa", "Egypt"]
    for i in codes:
        y = list(l.filter(like=i, axis=0)["Daily confirmed cases (cases)"])
        if i in countries:
            result = waipy.cwt(y, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, 'DOG', "x")
            waipy.wavelet_plot(i, range(len(y)), y, 0.03125, result, savefig=True)
        if len(y) > 50:
            SOC(y, i)
            alfa, xdfa, ydfa, reta = statsfuncs.dfa1d(y, 1)
            freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = statsfuncs.psd(y)
            values.append([statsfuncs.variance(y), statsfuncs.skewness(y), statsfuncs.kurtosis(y), alfa, index, mfdfa.makemfdfa(y), i])

    skew2 = []
    alfa = []
    kurt = []
    index = []
    psi = []

    for i in range(len(values)):
        skew2.append(values[i][1] ** 2)
        kurt.append(values[i][2])
        alfa.append(values[i][3])
        index.append(values[i][6])

    skew2 = np.array(skew2)
    alfa = np.array(alfa)
    kurt = np.array(kurt)

    kk = pd.DataFrame({'Skew²': skew2, 'Alpha': alfa})
    K = 20
    model1 = KMeans()
    visualizer = KElbowVisualizer(model1, k=(1, K))
    kIdx = visualizer.fit(kk)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
    kIdx = kIdx.elbow_value_
    model1 = KMeans(n_clusters=kIdx).fit(kk)
    # scatter plot
    ax = plt.figure()
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0, kIdx):
        ind = (model1.labels_ == i)
        plt.scatter(skew2[ind], alfa[ind], s=30, c=clr[i], label='Cluster %d' % i)

    plt.xlabel("Skew²")
    plt.ylabel("Alfa")
    plt.title('KMeans clustering with K=%d' % kIdx)
    plt.legend()
    plt.show()

    kk = pd.DataFrame({'Skew²': skew2, 'Alpha': alfa, 'Cluster skew²': model1.labels_}, index=index)
    kk = kk.sort_values(by=["Cluster skew²"])
    kk.to_csv("sort_by_skew.csv")

    kk = pd.DataFrame({'Kurtosis': kurt, 'Alpha': alfa})
    K = 20
    model2 = KMeans()
    visualizer = KElbowVisualizer(model2, k=(1, K))
    kIdx = visualizer.fit(kk)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
    kIdx = kIdx.elbow_value_
    model2 = KMeans(n_clusters=kIdx).fit(kk)
    # scatter plot
    ax = plt.figure()
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0, kIdx):
        ind = (model2.labels_ == i)
        plt.scatter(kurt[ind], alfa[ind], s=30, c=clr[i], label='Cluster %d' % i)

    plt.xlabel("Kurtosis")
    plt.ylabel("Alfa")
    plt.title('KMeans clustering with K=%d' % kIdx)
    plt.legend()
    plt.show()

    kk = pd.DataFrame({'Kurt': kurt, 'Alpha': alfa, 'Cluster kurt': model2.labels_}, index=index)
    kk = kk.sort_values(by=["Cluster kurt"])
    kk.to_csv("sort_by_kurt.csv")

    skew2 = []
    alfa = []
    kurt = []
    index = []
    psi = []

    for i in range(len(values)):
        if not np.isnan(values[i][5]):
            skew2.append(values[i][1] ** 2)
            psi.append(values[i][5])
            index.append(values[i][6])
        else:
            print("Excluded country: {}".format(values[i][6]))

    skew2 = np.array(skew2)
    psi = np.array(psi)

    kk = pd.DataFrame({'Skew²': skew2, 'Psi': psi})
    K = 20
    model3 = KMeans()
    visualizer = KElbowVisualizer(model3, k=(1, K))
    kIdx = visualizer.fit(kk)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
    kIdx = kIdx.elbow_value_
    model3 = KMeans(n_clusters=kIdx).fit(kk)
    # scatter plot
    ax = plt.figure()
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0, kIdx):
        ind = (model3.labels_ == i)
        plt.scatter(skew2[ind], psi[ind], s=30, c=clr[i], label='Cluster %d' % i)

    plt.xlabel("Skew²")
    plt.ylabel("Psi")
    plt.title('KMeans clustering with K=%d' % kIdx)
    plt.legend()
    plt.show()

    kk = pd.DataFrame({'Skew²': skew2, 'Psi': psi, 'Cluster psi': model3.labels_}, index=index)
    kk = kk.sort_values(by=["Cluster psi"])
    kk.to_csv("sort_by_psi.csv")

def main():
    dado = criarDataSet()
    dadoRegional = criarDataSetRegional()

    obterVisualizacao(dado, dadoRegional)
    obterHistograma(dado, dadoRegional)
    plotCullenFrey(dado, dadoRegional)
    obterPDF(dado, dadoRegional)
    obterRegressaoLinear(dado, dadoRegional)
    obterGandSCurve(dado, dadoRegional)
    obterSOC()

main()
