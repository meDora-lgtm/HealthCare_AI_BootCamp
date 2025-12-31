import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

iris = sns.load_dataset('iris')

X = iris.iloc[:, 0].values.reshape(-1, 1)
y = iris.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# sepal_length를 가지고 petal_length를 추론하는 과정 중 데이터셋을 8:2로 나눠줌
# random_state 숫자는 딱히 상관은 없지만 42로 많이 사용함, 랜덤 난수를 결정하는 숫자

model = LinearRegression()

model.fit(X_train, y_train)

w = model.coef_[0]
# w에 기울기(weight)를 등록

b = model.intercept_
# b에 y절편(편향)을 등록

y_pred = model.predict(X_test)
# 모델을 가지고 추론해보는 과정

# 오차를 계산해보자
# r2 score를 통해 예측을 해석해보자
mse = mean_squared_error(y_test, y_pred) #평균 제곱 오차
r2 = r2_score(y_test, y_pred) #0~1  사이 1에 가까울수록 좋다

# X의 최솟값, 최댓값으로 범위 설정
# linspace로 최솟값, 최댓값 사이에 점을 채워넣는 것, 점을 100개 찍어서 선처럼 보이도록 함
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

# 모든 feature를 사용해서 예측해보자
# 세번째 petal_length를 제외한 나머지 데이터를 X_all에 넣어야 함
X_all = iris.iloc[:, [0, 1, 3]].values
X_all.shape

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
model_all = LinearRegression()
# fit으로 학습 진행
model_all.fit(X_all_train, y_all_train)

y_all_pred = model_all.predict(X_all_test)

mse_all = mean_squared_error(y_all_test, y_all_pred)

# feature가 하나일 때와 비교했을 때 r2 score가 훨씬 오른 것을 확인할 수 있음
r2_all = r2_score(y_all_test, y_all_pred)

# 가중치도 바뀐 것을 확인할 수 있음
model_all.coef_

# 절편도 바뀐 것을 확인 가능(세개의 feature을 아우르는 값 하나만 나옴)
# 확인 결과 유의미한 feature가 많을수록 더 정확한 예측이 가능하다는 것을 알 수 있음
model_all.intercept_

# ===== 그래프를 3개 subplot으로 한 화면에 =====

plt.figure(figsize=(18, 5))

# 1) 단일 feature: train/test + 회귀선
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, label="train")
plt.scatter(X_test, y_test, label="test")
plt.plot(x_line, y_line, label="regression line")
plt.title("Single feature: sepal_length -> petal_length")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.legend()

# 2) 다중 feature: feature 0 vs y (train/test)
plt.subplot(1, 3, 2)
plt.scatter(X_all_train[:, 0], y_all_train, label="train")
plt.scatter(X_all_test[:, 0], y_all_test, label="test")
plt.title("Multi features: feature0 vs y")
plt.xlabel("feature0 (sepal_length)")
plt.ylabel("petal_length")
plt.legend()

# 3) 다중 feature: y_true vs y_pred
plt.subplot(1, 3, 3)
plt.scatter(y_all_test, y_all_pred)
plt.title("Multi features: y_true vs y_pred")
plt.xlabel("y_true")
plt.ylabel("y_pred")

plt.tight_layout()
plt.show()
