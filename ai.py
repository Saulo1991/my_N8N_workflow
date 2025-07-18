import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

default_test_size = 0.2

seed = 42

df = pd.read_excel('007-TabelaTACO.xlsx')

column_names = list(df)
column_names.remove('Número')
column_names.remove('Nome Alimento')

for names in column_names[1:]:
    df[names] = pd.to_numeric(df[names], errors = 'coerce')

df = df[["Proteína", "Lipídeos", "Carboidrato", "Energia_KCAL"]]
df = df.dropna()

print('Número de dados válidos: ', len(df.index))

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = default_test_size, random_state = seed)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('preprocessor', PolynomialFeatures())
    ('regressor', LinearRegression())
])

# hyperparameters grid to search within
hyperparameters = {'preprocessor__degree': [1, 2, 3]}

grid_search = GridSearchCV(pipe,
                           param_grid=hyperparameters,
                           return_train_score=True,
                           scoring='neg_mean_squared_error',
                           n_jobs=-2,
                           cv = 5)

grid_search.fit(X_train, y_train)

# Print Best Hyperparameters
cv_best_params = grid_search.best_params_
print('\n Best hyperparameters:')
print(cv_best_params)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

pipe.set_params(**cv_best_params)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

pipe['regressor'].coef_

pipe['regressor'].intercept_

column_names = list(df)
i = 1
for names in column_names[:-1]:
    InputString = []
    for j in range(len(column_names) - 1):
        if column_names[j] == names:
            InputString.append(100)
        else:
            InputString.append(0)
    print(names, pipe.predict([InputString])/100)
    i = i + 1

print('Protein Atwater general factors', 16.7/4.184)
print('Fat Atwater general factors', 37.4/4.184)
print('Carbohydrate Atwater general factors', 16.7/4.184)

# Análisse dos erros das previsões

rmse_test = math.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)
mape_test = mean+mean_absolute_percentage_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print('RSME | MAE | MAPE | R2')
print(f'{round(rmse_test,4)} | {round(mae_test,4)} | {round(mape_test,4)} | {round(r2_test, 4)}')

y_pred = pd.DataFrame(data=pipe.predict(X_test), columns=['Predicted Values'])

y_real = pd.DataFrame(data=y_test, columns=['Real Values'])

df_comparison = pd.concat([y_real, y_pred], axis=1)
df_comparison.columns = ['Real_Data', 'Predicted Value']
df_comparison['Percentage_difference'] = 100*(df_comparison['Predicted_Value'] - df_comparison['Real_Data']) / df_comparison['Real_Data']
df_comparison['Average'] = df_comparison['Real Data'].mean()
df_comparison['Q1'] = df_comparison['Real_Data'].quantile(0.25)
df_comparison['Q3'] = df_comparison['Real_Data'].quantile(0.75)
df_comparison['USL'] = df_comparison['Real Data'].mean() + 2*df_comparison['Real Data'].std()
df_comparison['LSL'] = df_comparison['Real Data'].mean() - 2*df_comparison['Real Data'].std()

df_comparison.sort_index(inplace=True)

df_comparison

df_comparison.describe()

# Graphic visualization of predictions by real values
plt.figure(figsize=(25,10))
plt.title('Real Value vs Predicted Value', fontsize=25)
plt.plot(df_comparison.index, df_comparison['Real_Data'], label = 'Real', marker='D', markersize=10, linewidth=0)
plt.plot(df_comparison.index, df_comparison['Predicted_Value'], label = 'Predicted', c='r', linewidth=1.5)
plt.plot(df_comparison.index, df_comparison['Average'], label = 'Mean', linestyle='dashed', c='yellow')
plt.plot(df_comparison.index, df_comparison['Q1'], label = 'Q1', linestyle='dashed', c='b')
plt.plot(df_comparison.index, df_comparison['Q3'], label = 'Q3', linestyle='dashed', c='g')
plt.plot(df_comparison.index, df_comparison['USL'], label = 'USL', linestyle='dashed',c='r')
plt.plot(df_comparison.index, df_comparison['LSL'], label = 'LSL', linestyle='dashed',c='r')

plt.legend(loc='best')
plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# Graphic visualization of predictions by real values
plt.figure(figsize=(25,10))
plt.title('Real Value vs Predicted Value', fontsize=25)
plt.scatter(df_comparison['Real_Data'], df_comparison['Predicted_Value'], s=100)
plt.plot(df_comparison['Real_Data'], df_comparison['Real_Data'], c='r')

plt.xlabel('Real', fontsize=25)
plt.ylabel('Predicted', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()


