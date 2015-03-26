#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 検証環境
# - Python 2.7.6
# - pip 1.5.4
#   - pandas 0.15.2
#   - scikit-learn 0.15.2
#   - numpy 1.9.1
#   - scipy 0.13.3

# モジュールを読み込む
import re
import math
import pandas as pd
from sklearn import linear_model

##################################
# データごにょごにょするための関数
##################################
# 日付から曜日だけ抽出し, '平日' or '土日祝日' のどちらかの文字列を返す
def extract_youbi(s):
    pat = re.compile('\d+/\d+\(([^\)]+)\)')
    youbi = pat.match(s.decode("utf-8")).group(1).encode("utf-8")
    if '祝' in youbi or '休' in youbi or '日' in youbi or '土' in youbi:
        return '土日祝休'
    else:
        return '平日'

# 節から曜日だけ抽出し, '平日' or '土日祝日' のどちらかの文字列を返す
def extract_turn(s):
    pat = re.compile(u'.*?節')
    return pat.match(s.decode("utf-8")).group().encode("utf-8")

########################
# データの読み込み処理
########################
# 学習用データの読み込み
df_train_2010 = pd.read_csv("data/train-2010.csv", header = None)
df_train_2011 = pd.read_csv("data/train-2011.csv", header = None)
df_train = pd.read_csv("data/train.csv", header = None)

df_train_2010.columns = df_train_2011.columns = df_train.columns = ["ID", "passenger", "year", "league", "turn", "date", "time", "home", "away", "stadium", "TV"]

# 学習用データをマージ
df_train = pd.concat([df_train_2010, df_train_2011, df_train])

# 予測用データを読み込む
df_test = pd.read_csv("data/test.csv", header = None)
df_test.columns = ["ID", "year", "league", "turn", "date", "time", "home", "away", "stadium", "TV"]

# 学習用データと予測用データを結合する
len_train = len(df_train)
df = pd.concat([df_train, df_test])

##################################
# データを分析用にごにょごにょする
##################################
stadium = pd.get_dummies(df.stadium)
league = pd.get_dummies(df.league)
youbi = pd.get_dummies(df.date.map(extract_youbi))
turn = pd.get_dummies(df.turn.map(extract_turn))
month = pd.get_dummies(df.date.map(lambda d: d.decode('utf-8')[:2].encode('utf_8'))) #月だけ抜き出す
away = pd.get_dummies(df.away)

########################
# 学習に使う特徴量を選択
########################
f_values = (stadium, league, youbi, turn, month, away)
df_dummy = pd.concat(f_values, axis = 1)
variables = list(df_dummy.columns)

# モデル用のデータを加工する
df_x_train = df_dummy[df_dummy.keys()][:len_train]
df_x_test = df_dummy[df_dummy.keys()][len_train:]
df_y_train = df_train.passenger

####################
# モデルに当てはめる
####################
df_result = pd.read_csv("data/sample.csv", header = None)
clf = linear_model.LassoLarsIC()
clf.fit(df_x_train, df_y_train)
predict = clf.predict(df_x_test)
df_result[1] = predict
df_result.to_csv("submission.csv", index = None, header = None)

# print "パラメータ"
# print clf.get_params()
# print "決定関数の独立項: %s" % clf.coef_
# print "切片: %s" %  clf.intercept_
# print "alpha: %s" %  clf.alpha_
# print "各変数のモデルへの寄与度"
# for i in range(len(variables)):
  # print "%s: %s" % (variables[i], clf.coef_[i])
