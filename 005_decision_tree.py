# Get important columns
# importance_threshold = 9
# important_columns = []
# for i, x in zip((feature_score['Score'].round(2) > importance_threshold).index, (feature_score['Score'].round(2) > importance_threshold)):
#     if x:
#         important_columns.append(X.columns[i])
# print("important_columns:", important_columns)


temp_df_new = temp_df[temp_df["position"] == "subset1"].drop(["position", "range", "top_range"], axis=1) # [["rank1", "my_position_ip", "target"]]

train, test = train_test_split(temp_df_new, test_size=0.40)


train_X = train.drop(["target"], axis=1)
train_y = train[["target"]]
test_X = test.drop(["target"], axis=1)
test_y = test[["target"]]
min_samples_split = int(train_X.shape[0]*0.3)


from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import tree, linear_model
%matplotlib inline
for i in range(1,7):
    clf2 = tree.DecisionTreeRegressor(max_depth=i, min_samples_split=min_samples_split)
    clf2.fit(train_X, train_y)
    pred_y = clf2.predict(test_X)
    from sklearn.metrics import mean_squared_error
    score = np.sqrt(mean_squared_error(pred_y, test_y))
    print("score:", score)
    # clf2.coef_

CHOICE = 5

clf2 = tree.DecisionTreeRegressor(max_depth=CHOICE, min_samples_split=min_samples_split)
clf2.fit(train_X, train_y)
pred_y = clf2.predict(train_X)
from sklearn.metrics import mean_squared_error
print("score:", score)
dot_data = export_graphviz(clf2, out_file=None, feature_names=train_X.columns)
graph = Source(dot_data)
graph.render("graph")
graph
# clf2.coef_
