%matplotlib inline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score, log_loss, precision_score
from pandas import DataFrame, concat
from numpy.random import seed
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from catboost import Pool
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.preprocessing import PolynomialFeatures
import shap
import datetime
import matplotlib.pyplot as plt
from numpy import max
from pandasql import sqldf
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
experiment_model = CatBoostRegressor()
df = pd.read_csv("../data/interim/train.csv")


seed(42)
# temp_df = df.copy()
temp_df = df.sample(100000)

# X = temp_df.drop(["target", "position"], axis=1)
X = temp_df.drop(["target"], axis=1)
y = temp_df[["target"]]


# X = X[important_columns]
cat_index = []
for i, col zip(range(X.shape[1]), X.columns):
    if X[col].dtype == 'str':
        cat_index.append(i)
cat_features = cat_index  # Index of each category column

# Get shap dataframes
X = DataFrame(X, columns=X.columns)

# Fit the model on the training set
experiment_model.fit(X, y, cat_features=cat_features, logging_level='Silent')

feature_score = pd.DataFrame(list(zip(X.dtypes.index, experiment_model.get_feature_importance(Pool(X, label=y, cat_features=cat_features)))),
                columns=['Feature','Score'])

feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()
