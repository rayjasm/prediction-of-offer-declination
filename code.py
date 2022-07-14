from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd

data= pd.read_csv('drive/My Drive/data.csv',encoding="cp932")

# 行数、列数表示
print(data.shape)
# 欠損値埋める
df = data.fillna("null")
# サイズ表示
df.groupby(df["07"]).size()
df.head()

# 不要な行削除
del df["name"]
del df["mobile"]

# one-hot encoding
df_1 = pd.get_dummies(df[["01","02","03"]])

# 採用したい人の中で辞退した人にフラグをつける
# 辞退した人は1、それ以外は0
df['flg'] = 0
df.loc[df["07"] == "辞退(1次選考中)", 'flg'] = '1'
df.loc[df["07"] == "辞退(2次選考中)", 'flg'] = '1'
df.loc[df["07"] == "辞退(入社)", 'flg'] = '1'
df.loc[df["07"] == "辞退(内定)", 'flg'] = '1'
df.loc[df["07"] == "辞退(受付中)", 'flg'] = '1'
df.loc[df["07"] == "辞退(書類選考中)", 'flg'] = '1'
df.loc[df["07"] == "辞退(最終選考中)", 'flg'] = '1'
df.loc[df["07"] == "辞退(面接日決定後)", 'flg'] = '1'
df.loc[df["07"] == "連絡不通", 'flg'] = '1'

df_1['decline'] = df['flg']
df_1.head()

# 型を直しとく
df_1['decline'] = df_1['decline'].astype(int)
print(format(df_1.dtypes))

first_column = df_1.pop('decline')
df_1.insert(0,'decline',first_column)

# 相関関係を調べる
df_1.corr()

# とりあえず使うやつ
print(df_1.shape)
a = df_1[0:1]
a = a.drop('decline', axis=1)
a.head()

# SVM
import numpy as np
from sklearn import svm

# 説明変数と目的変数の設定
X = df_1.drop("decline", axis=1)
y = df_1["decline"]

model = svm.SVC()
model.fit(X,y)

decline = a
ans = model.predict(decline)

if ans == 0:
    print("辞退")
if ans == 1:
    print("辞退以外")


# 回帰分析
# データ分割のためのインポート
from sklearn.model_selection import train_test_split
# モデル構築のためのインポート
from sklearn.linear_model import LinearRegression

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

X = df_1.drop("decline", axis=1)
y = df_1["decline"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 重回帰分析は過学習になったのでリッジ回帰
clf = linear_model.Ridge()
clf.fit(X_train, y_train)

# 回帰係数と切片の抽出
a = clf.coef_
b = clf.intercept_  

# 回帰係数
print("回帰係数:", a)
print("切片:", b) 
print("決定係数(train):", clf.score(X_train, y_train))
print("決定係数(test):", clf.score(X_test, y_test))


# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df_1.drop("decline", axis=1)
y = df_1['decline']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

print('train:{:.3}'.format(model.score(X_train, y_train)))
print('test:{:.3}'.format(model.score(X_test, y_test)))

# ダウンロード用
from google.colab import files
data.to_csv('saiyou.csv', encoding='utf_8_sig', index=False)
files.download('result.csv')