clf_models = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "LogisticRegression",
    "SVC",
    "KNeighborsClassifier",
    "MLPClassifier",
    "AdaBoostClassifier",
    "VotingClassifier",
    "StackingClassifier",
]

clf_metrics = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "precision_recall_auc",
]

regress_models = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "DecisionTreeRegressor",
    "LinearRegression",
    "Lasso",
    "ElasticNet",
    "SVR",
    "KNeighborsRegressor",
    "AdaBoostRegressor",
    "VotingRegressor",
    "StackingRegressor",
]

regress_metrics = ["mse", "rmse", "mae", "mape", "r2"]
