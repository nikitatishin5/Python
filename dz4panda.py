import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/Grossmend/CSV/master/titanic/data.csv', nrows=17, usecols=['Name', 'Sex', 'Survived'])
print(df)
df_each = pd.DataFrame()
for chunk in df:
    df_each = df_each.append(chunk.iloc[0,:])
print(df_each)


test_list = 'jsfbsjdhfbsdjhfbsfjsdhbfidshbewrew'
test_arr = np.arange(len(test_list))
test_dict = dict(zip(test_list, test_arr))
s = pd.Series(test_dict)
# сбрасываем индексы
df = s.to_frame().reset_index()
df.columns=['letter', 'number']
print(df)


df = pd.read_csv('https://raw.githubusercontent.com/Grossmend/CSV/master/titanic/data.csv', nrows=10)


print('\n', 'Формат столбцов:')
print(df.dtypes)
print('\n', 'Размерность:')
print(df.shape)
print('\n', 'Общая статистика')
print(df.describe())

