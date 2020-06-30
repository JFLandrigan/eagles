import sys

if "win" in sys.platform:
    ext_char = "\\"
else:
    ext_char = "/"

clf_models = [
    "RandomForestClassifier",
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

regress_models = [
    "RandomForestRegressor",
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
