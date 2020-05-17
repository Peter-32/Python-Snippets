# Get important columns
importance_threshold = 7
important_columns = []
for i, x in zip((feature_score['Score'].round(2) > importance_threshold).index, (feature_score['Score'].round(2) > importance_threshold)):
    if x:
        important_columns.append(X.columns[i])
print("important_columns:", important_columns)


import numpy as np
from pandas import read_csv
from numpy.random import seed
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
warnings.filterwarnings("ignore")

import shap
import pandas as pd
from numpy import cumsum
from xgboost import XGBClassifier

seed(40)

# shap values
shap_values = shap.TreeExplainer(experiment_model).shap_values(X[0:10000])

sorted_feature_importance = pd.DataFrame(shap_values, columns=X.columns).abs().sum().sort_values(ascending=False)
cumulative_sum = cumsum([y for (x,y) in sorted_feature_importance.reset_index().values])
gt_999_importance = cumulative_sum / cumulative_sum[-1] > .999
nth_feature = min([y for (x,y) in zip(gt_999_importance, zip(range(len(gt_999_importance)))) if x])[0]
important_columns = sorted_feature_importance.iloc[0:nth_feature+1].index.values.tolist()
print(important_columns)

plt.clf()
shap.summary_plot(shap_values, X[0:10000])
