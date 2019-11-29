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
