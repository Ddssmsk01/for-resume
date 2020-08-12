import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('F:/for resume/adult.data', header=None, names=columns, na_values=' ?')
df = df.drop('education', axis=1)
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
df = df.dropna()

test = pd.read_csv('F:/for resume/adult.test', header=None, names=columns, na_values=' ?', skiprows=1)
test = test.drop('education', axis=1)
test['income'] = test['income'].map({' <=50K.': 0, ' >50K.': 1})
test = test.dropna()

X_train = pd.get_dummies(df).drop('income', axis=1)
y_train = df['income']


X_test = pd.get_dummies(test).drop('income', axis=1)
y_test = test['income']

columns = set(X_train.columns) | set(X_test.columns)
X_train = X_train.reindex(columns=columns).fillna(0)
X_test = X_test.reindex(columns=columns).fillna(0)

from sklearn.metrics import classification_report


from xgboost import XGBClassifier
model = XGBClassifier(seed=42,
                      n_estimators=100,
                      max_depth=6,
                      learning_rate=0.3)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
print (classification_report(y_train, y_pred_train))
y_pred = model.predict(X_test)
print (classification_report(y_test, y_pred))

import seaborn as sns
sns.set(font_scale = 1.5)

import xgboost as xgb
xgb.plot_importance(model)
xgb.plot_importance(model, max_num_features = 30)
plt.show()

from sklearn.model_selection import GridSearchCV
grid_param = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.3]
}
zzz = GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1,
       colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=1234, silent=True, subsample=0.8),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)
zzz.fit(X_train, y_train)

sorted(zzz.cv_results_.keys())
print(zzz.grid_scores_)
