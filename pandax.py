import pandas as pd

df = pd.DataFrame([
            [1, 'Saulo', 0.4],
            [177, 8, 'Moreira']],
            index=[45, 'Rosí'],
            columns=['Col1', 'Col2', 'col3'])

print(df.dtypes)

# Base de Dados - COVID Global

link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notifications = pd.read_csv(link)
print(df_notifications)

# ____________________________________________________________________________________________


link2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link2)
df_notification.drop(['Lat', 'Long'], axis=1, inplace=True)
print(df_notification)

# ____________________________________________________________________________________________

link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
df_notification.drop(['Country/Region', 'Long'], axis=1, inplace=True)
print(df_notification)


#Loc

link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
df_notification.loc[0]

link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
df_notification.loc[0, 'Country/Region']
print(df_notification.loc[:, 'Country/Region'])


link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
df_notification.loc[0, 'Country/Region']
print(df_notification.loc[54:87, ['Province/State', 'Country/Region', '12/14/21']])

#loc com condicional

link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
df_notification['Country/Region'] == 'Brazil'


link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/refs/heads/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_notification = pd.read_csv(link)
print(df_notification.loc[df_notification['12/14/21'] < 1e06].head(5))