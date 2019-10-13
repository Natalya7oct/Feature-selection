#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# In[24]:


# Создаём сэмпл
# Сумма кредита зависит от ЗП и от фактических трат и не зависит от возраста клиента
n_samples = 500

age = np.random.choice(100, n_samples) + 21
salary = np.random.choice(150, n_samples) + 70
spending = np.random.choice(200, n_samples) + 10

credit = spending/salary*50+10

data = pd.DataFrame({'age': age, 'salary': salary, 'spending': spending, 'credit': credit})
data.head(5)


# In[25]:


from sklearn.metrics import mean_absolute_error

X = data[['age', 'salary', 'spending']]
y = data['credit']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['age', 'salary', 'spending']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))


# Видим, что вес у возраста совсем маленький, незначительный. 

# In[26]:


X = data[['spending', 'salary']]
y = data['credit']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['spending', 'salary']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))


# Возраст убрали, но средняя абсолютная ошибка не изменилась.

# In[27]:


# Создаем новый признак
data['mult'] = data['spending']/data['salary']
data.head(5)


# In[28]:


X = data[['mult']]
y = data['credit']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['mult']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))


# После подбора признака модель стала гораздо точнее.

# In[ ]:




