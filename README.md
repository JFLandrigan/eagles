# eagles

Creator: Jon-Frederick Landrigan



## Description:

This repository contains utilities to perform tasks relating to data science 
including supervised and unsupervised machine learning, data exploration and statistical testing.
The functions primarily act as utility wrappers.

For examples of how to used the functions contained within the package see the following jupyter notebooks:
- Supervised Tuning.ipynb
- Unsupervised Tuning.ipynb



## Install and How to use it?
**This package is still under heavy development and testing. ** Currently the package is only available for installation via github. To install you can use ```pip3 install git+https://github.com/JFLandrigan/eagles/tree/master ```  .  Once installed it can be imported like any other python package. 



## Supported Model Abbreviations

Note that the functions primarily support sklearn model objects however if a model follows the standard fit, predict and predict_proba methods it can be passed in directly as well.

| Classification                         | Regression                                |
| -------------------------------------- | ----------------------------------------- |
| "rf_clf" : RandomForestClassifier      | "rf_regress" : RandomForestRegressor      |
| "gbc_clf" : GradientBoostingClassifier | "gbc_regress" : GradientBoostingRegressor |
| "dt_clf" : DecisionTreeClassifier      | "dt_regress" : DecisionTreeRegressor      |
| "logistic" : LogisticRegression        | "linear" : LinearRegression               |
| "svc" : SVC                            | "lasso" : Lasso                           |
| "knn_clf" : KNeighborsClassifier       | "elastic" : ElasticNet                    |
| "nn" : MLPClassifier                   | "svr" : SVR                               |
| "ada_clf" : AdaBoostClassifier         | "knn_regress" : KNeighborsRegressor       |
|                                        | "ada_regress" : AdaBoostRegressor         |



## Metric Options

| Classification                                               | Regression                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 'accuracy'                                                   | 'mse' - mean square error                                    |
| 'f1'                                                         | 'rmse' - root mean square error                              |
| 'precision'                                                  | 'mae' - mean absolute error                                  |
| 'recall'                                                     | 'mape' - mean absolute percent error (note zeros are excluded) |
| 'roc_auc' - Area Under the Receiver Operating Characteristic Curve | 'r2'                                                         |
| 'precision_recall_auc' - Area Under the Precision Recall Curve |                                                              |



## tune_test_model() parameter search options

- 'random_cv' : sklearn implementation of random parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'bayes_cv' :  scikit-optimize implementation of bayes parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'grid_cv' : sklearn implementation of paramter grid search. Can pass in n_jobs to parallelize fits.



## How to pass in parameters for tune_test_model()

- Single model: pass in relative parameters as a dictionary with key (parameter) value (listed parameter setting) pairs

- Model embedded within a pipeline: pass in relative parameters as dictionary  with key (parameter) value (listed parameter setting) pairs. Note that the parameter keys should follow the format  ```clf__< parameter >```

- Models embedded within a VotingClassifier or Voting Regressor: pass in relative parameters as dictionary  with key (parameter) value (listed parameter setting) pairs. Note that the parameter keys should follow the format (note the follwing example assumes a random forest and a logistic regression):

  ```
  pars = {'rf__clf__n_estimators':[x for x in range(100,300,50)]
          ,'rf__clf__max_depth':[5, 10, 20]
          ,'rf__clf__class_weight': [None, 'balanced_subsample']
  
          ,'lr__clf__penalty': ['l1','l2']
          ,'lr__clf__class_weight':[None, 'balanced']
          ,'lr__clf__max_iter':[100, 1000]
          ,'lr__clf__C': [.25,.5,1]
  
          ,'weights':[[1,1],[1,2],[2,1]] 
         }
  ```



## Passing in Pipelines

Sklearn pipeline objects can be passed directly into supervised_tuner.model_eval() and supervised_tuner.tune_test_model() via the model argument (i.e. when a model is embedded within a pipeline) or the pipe argument. When the pipeline is passed into the pipe argument the model will be appended and/or embedded within the passed in pipeline. Note the following conventions for pipeline prefixes should be followed:

- 'impute' : for imputation steps
- 'scale': for scaling steps
- 'clf' : for the classifiers

See the following pipeline construction examples

```
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

lr_pipe = Pipeline([('impute', SimpleImputer(strategy='median'))
                    ,('scale', MinMaxScaler())
                    ,('clf', LogisticRegression())])
```



## Logging

The supervised_tuner model_eval() and tune_test_model() allow for logging of the models being tested.
Note that is no log path is passed in a data subdirectory will be created in eagles/eagles/Supervised/utils/



## Notes

Currently the functions primarily rely on the use of pandas data frames. Numpy matrices can be passed in
however this may result in unexpected behavior. 



## Packages Required (see requirements.txt for versions)
- gensim
- imbalanced-learn
- kneed
- matplotlib
- nltk
- numpy
- pandas
- pingouin
- scikit-learn
- scikit-optimize
- scipy
- seaborn
- statsmodels