#1
import sys
import numpy as np
import pandas as pd

#2
src_list = list('abcde')
src_arr = np.arange(5)
src_dict = dict(zip(src_list, src_arr))

s1 = pd.Series(src_list)
s2 = pd.Series(src_arr)
s3 = pd.Series(src_dict)

print(s1)
print(s2)
print(s3)

#3_4_5
s = pd.Series({'a': 'bear', 'b': 'meet', 'c': 'bmw'})
s = s.to_frame()
print(s)

#6
s1 = pd.Series(list('abcdefghij'))
s2 = pd.Series(np.arange(10))

df = pd.concat([s1, s2], axis=1)
print(df)

#7
s1 = pd.Series(list('abcdefghij'))
s1.name = 'my_name'
print(s1)

#8_9_10
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([4, 5, 6, 7, 8])
s3 = s1[~s1.isin(s2)]
print(s3)
s3 = np.setdiff1d(s1, s2, assume_unique=False)
print(s3)


#11-16
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([4, 5, 6, 7, 8])

s1 = pd.Series(np.union1d(s1, s2)) #не повторяются
s2 = pd.Series(np.intersect1d(s1, s2)) #пересеченные данные
s3 = s1[~s1.isin(s2)] #все кто не пересекаются
print(s3)
s3 = np.setxor1d(s1, s2, assume_unique=False)
print(s3)

#ДЗ_2

print("DZ_2")

#Получить от объекта Series показатели описательной статистики

state = np.random.RandomState(42)

s = pd.Series(state.normal(10, 5, 25))
pkz = s.describe()
print(pkz)

#Узнать частоту уникальных элементов объекта Series (гистограмма)
a = 'abcdefghik'
len_series = 30
s = pd.Series(np.take(list(a), np.random.randint(len(a), size=len_series)))

ans = s.value_counts()

print(ans)

#Заменить все элементы объекта Series на "Other", кроме двух наиболее часто встречающихся
state = np.random.RandomState(42)
s = pd.Series(state.randint(low=1, high=5, size=[13]))
print(s.value_counts())
s[~s.isin(s.value_counts().index[:2])] = 'Other'
print(s)

#Создать объект Series в индексах дата каждый день 2019 года, в значениях случайное значениеНайти сумму всех вторников
#Для каждого месяца найти среднее значение

dti = pd.date_range(start='2019-01-01', end='2019-12-31', freq='B')
s = pd.Series(np.random.rand(len(dti)), index=dti)

ans1 = s[s.index.weekday == 2].sum()
print('Сумма всех "вторников"', ans1)
print()

ans2 = s.resample('M').mean()
print('Средние значения по месяцам:\n', ans2)
print()

#Преобразовать объект Series в DataFrame заданной формы (shape)
s = pd.Series(np.random.randint(low=1, high=10, size=[35]))
r = (7, 5)

if r[0] * r[1] != len(s):
    sys.exit('не возможно применить reshape')

df = pd.DataFrame(s.values.reshape(r))

print(df)

#Найти индексы объекта Series кратные 3
s = pd.Series(np.random.randint(low=1, high=10, size=[7]))

ans = s[s % 3 == 0].index
print(ans)

#Получить данные по индексам объекта Series

s = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
p = [0, 4, 8, 14, 20, 10]
ans = s[p]
print(ans)

#Объединить два объекта Series вертикально и горизонтально
s1 = pd.Series(range(5))
s2 = pd.Series(list('abcde'))

ans_vertical = s1.append(s2)
ans_horizontal = pd.concat([s1, s2], axis=1)

print(ans_vertical)
print(ans_horizontal)

#Получить индексы объекта Series A, данные которых содержатся в объетке Series B
s1 = pd.Series([5, 3, 2, 1, 4, 11, 13, 8, 7])
s2 = pd.Series([1, 5, 13, 2])
ans = np.asarray([np.where(i == s1)[0].tolist()[0] for i in s2])
print(ans)

#Получить объект Series B, котоырй содержит элементы без повторений объекта A
s = pd.Series(np.random.randint(low=1, high=10, size=[10]))
ans = pd.Series(s.unique())
print(ans)

#Преобразовать каждый символ объекта Series в верхний регистр_
# преобразовать данных Series в строку
s = pd.Series(['life', 'is', 'interesting'])
s = pd.Series(str(i) for i in s)
ans = s.map(lambda x: x.title())
print(ans)

#Рассчитать количество символов в объекте Series
# преобразовать в строковый тип
s = pd.Series(['one', 'two', 'three', 'four', 'five'])
s = pd.Series(str(i) for i in s)
ans = np.asarray([len(i) for i in s])
print(ans)

#Найти разность между объектом Series и смещением объекта Series на n
n = 1
s = pd.Series([1, 5, 7, 8, 12, 15, 17])
ans = s.diff(periods=n)
print(ans)

#Преобразовать разыне форматы строк объекта Series в дату
s = pd.Series(['2019/01/01', '2019-02-02', '15 Jan 1678'])
ans = pd.to_datetime(s)
print(ans)

#Поскольку работа с датой часто встречается в работе, то см. еще один пример
# все данные должны иметь одинаковый формат (часто бывает выгрузка из SQL)
s = pd.Series(['14.02.2019', '22.01.2019', '01.03.2019'])
# преобразование в дату
ans = pd.to_datetime(s, format='%d.%m.%Y')
print(ans)

#Получить год, месяц, день, день недели, номер дня в году от объекта Series (string)
from dateutil.parser import parse

s = pd.Series(['01 Jan 2019', '02-02-2088', '16870506'])
s_ts = s.map(lambda x: parse(x, yearfirst=True))
print(s_ts.dt.year)
print(s_ts.dt.month)
print(s_ts.dt.day)
print(s_ts.dt.weekofyear)
print(s_ts.dt.dayofyear)

#Отобрать элементы объекта Series, кторые содержат не менее двух гласных

from collections import Counter

s = pd.Series(['Мандарин', 'Bomb', 'Meat', 'Сосиска', 'Сосна', 'Table', 'VrUnityLeapMotion'])
mask = s.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiouаоиеёэыуюя')]) >= 2)
ans = s[mask]
print(ans)

#Отобрать e-маилы из объекта Series (можно юзать регулярки)
import re

emails = pd.Series(['commommailMAI @test.com', 'test@mail.ru', 'test.ToTo_ru', 'test@yamme'])
pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mask = emails.map(lambda x: bool(re.match(pattern, x)))
ans = emails[mask]
print(ans)

#Получить среднее значение каждого уникального объекта Series s1 через "маску" другого объекта Series s2
n = 10
s1 = pd.Series(np.random.choice(['dog', 'cat', 'horse', 'bird'], n))
s2 = pd.Series(np.linspace(1,n,n))
ans = s2.groupby(s1).mean()
print(ans)
