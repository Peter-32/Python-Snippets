from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(ohe.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(ohe.transform(X_valid[low_cardinality_cols]))

OH_cols_train.columns = ohe.get_feature_names(low_cardinality_cols)
OH_cols_valid.columns = ohe.get_feature_names(low_cardinality_cols)

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
