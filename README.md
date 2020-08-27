# eagles

Creator: Jon-Frederick Landrigan



## Description:

This repository contains utilities to perform tasks relating to data science 
including supervised and unsupervised machine learning, data exploration and statistical testing.
The functions primarily act as utility wrappers.

For examples of how to use the functions contained within the package see the following jupyter notebooks:
- Supervised Tuning.ipynb
- Unsupervised Tuning.ipynb
- Exploratory.ipynb



## Install
**This package is still under heavy development and testing.** Currently the package is only available for installation via github. To install you can use ```pip3 install git+https://github.com/JFLandrigan/eagles.git#"egg=eagles" ```  .  Once installed it can be imported like any other python package. For example:

```
from eagles.Supervised import supervised_tuner as st
from eagles.Unsupervised import unsupervised_tuner as ut
from eagles.Exploratory import explore, missing, distributions, categories , outcomes
```



## Supported Model Abbreviations

Note that the functions primarily support sklearn model objects however if a model follows the standard fit, predict and predict_proba methods it can be passed in directly as well.

### Supervised

| Classification                        | Regression                               |
| ------------------------------------- | ---------------------------------------- |
| "rf_clf" : RandomForestClassifier     | "rf_regress" : RandomForestRegressor     |
| "gb_clf" : GradientBoostingClassifier | "gb_regress" : GradientBoostingRegressor |
| "dt_clf" : DecisionTreeClassifier     | "dt_regress" : DecisionTreeRegressor     |
| "logistic" : LogisticRegression       | "linear" : LinearRegression              |
| "svc" : SVC                           | "lasso" : Lasso                          |
| "knn_clf" : KNeighborsClassifier      | "elastic" : ElasticNet                   |
| "nn" : MLPClassifier                  | "svr" : SVR                              |
| "ada_clf" : AdaBoostClassifier        | "knn_regress" : KNeighborsRegressor      |
| "et_clf": ExtraTreesClassifier        | "ada_regress" : AdaBoostRegressor        |
|                                       | "et_regress": ExtraTreesRegressor        |

### Unsupervised

Currently the functions primarily support the following the sklearn algorithms however other model objects can be passed in assuming they support the ```fit_predict()``` methodology like other sklearn clustering algorithms. 

- "kmeans"
- "agglomerativeclustering"
- "dbscan"



## Metric Options

### Supervised

When using ```supervised_tuner.tune_test_model()``` the tune_metric argument is used for the parameter search and the *eval_metric* argument is used for the final model evaluation (eval metrics should be passed in as a list). For ```supervised_tuner.model_eval()``` the metrics argument is used to tell the function what metrics to use (these should be passed in a list). If no metrics are passed in (for tuning and/or eval) classification problems will default to 'f1' and regression problems will default to 'mse'. Note that for multi-class classification problems the metrics default to "macro" averages. 

| Classification                                               | Regression                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 'accuracy'                                                   | 'mse' - mean square error                                    |
| 'f1'                                                         | 'rmse' - root mean square error                              |
| 'precision'                                                  | 'mae' - mean absolute error                                  |
| 'recall'                                                     | 'mape' - mean absolute percent error (note zeros are excluded) |
| 'roc_auc' - Area Under the Receiver Operating Characteristic Curve | 'r2' - r squared                                             |
| 'precision_recall_auc' - Area Under the Precision Recall Curve |                                                              |

### Unsupervised

When ```unsupervised_tuner.find_optimal_clusters()``` with K-Means or Agglomerative Clustering is used the following metrics can be used to find the "optimal" or "suggested "number of clusters however thorough analysis should be performed.

- "max_sil" : After generating models based on the range of cluster numbers desired the algorithm will pick the optimal number of clusters as the number of  clusters which resulted in the highest max silhouette score. 
- "knee_wss"  :  After generating models based on the range of cluster numbers desired the ```KneeLocator()``` method used (provided by the kneed package) to find the "elbow" or point at which increasing the number of clusters does not significantly decrease the amount of within cluster variability. 

Note the DBSCAN algorithm uses internal methods to find the optimal number of clusters. 



## supervised_tuner.tune_test_model() parameter search options

- 'random_cv' : sklearn implementation of random parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'bayes_cv' :  scikit-optimize implementation of bayes parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'grid_cv' : sklearn implementation of paramter grid search. Can pass in n_jobs to parallelize fits.



## How to pass in parameters for supervised_tuner.tune_test_model()

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

Sklearn pipeline objects can be passed directly into ```supervised_tuner.model_eval()``` and ```supervised_tuner.tune_test_model()``` via the model argument (i.e. when a model is embedded within a pipeline) or the pipe argument. When the pipeline is passed into the pipe argument the model will be appended and/or embedded within the passed in pipeline. Note the following conventions for pipeline prefixes should be followed:

- 'impute' : for imputation steps
- 'feature_selection' : for feature selection steps (using sklearn feature selection select from model)
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

The supervised_tuner ```model_eval()``` allows the used to log out a text file of the models performance and ```tune_test_model()``` allows the user to log out a text file containing relevant information for the call (e.g. tuning parameters, and model performance), the final model object and the data used for training and testing. ```model_eval()``` does not currently allow for data and model logging. See the following examples for argument definitions: 

```
# Model evaluation
st.model_eval(
    X=iris[fts],
    y=iris['dummy'],
    model='logistic',
    params={'solver':'liblinear'},
    metrics=["accuracy", "f1", "roc_auc"],
    bins=None,
    pipe=None,
    scale=None,
    num_top_fts=None,
    num_cv=5,
    get_ft_imp=True,
    random_seed=4,
    binary=True,
    log="log",
    log_name="model_eval_test.txt",
    log_path=None,
    log_note="This is a test of the model eval function"
)

# Tuning and testing a model (Note if only a log is wanted the argument can be set to 'log')
clf, params, clf_fts = st.tune_test_model(X=iris[fts]
                                            ,y=iris['dummy']
                                            ,model=vc_clf
                                            ,params=pars
                                            ,tune_metric="f1"
                                            ,eval_metrics=["accuracy", "f1", "precision"]
                                            ,num_cv=5
                                            ,pipe=None
                                            ,scale=None
                                            ,select_features=select_features_dict
                                            ,bins=None
                                            ,num_top_fts=None
                                            ,tuner="random_cv"
                                            ,n_iterations=15
                                            ,get_ft_imp=True
                                            ,n_jobs=2
                                            ,random_seed=None
                                            ,binary=True
                                            ,log=["log","mod","data"]
                                            ,log_name="model_tunetest_test.txt"
                                            ,log_path=None
                                            ,log_note=note)
```

**Note that if no log path is passed in a data subdirectory will be created in eagles/eagles/Supervised/utils/**



## Exploratory Module

The Exploratory module contains functions and tools for performing exploratory data analysis on pandas data frames. Currently the module includes the following:

- explore: Includes run_battery(), get_base_descriptives() and get_correlations()
  - run_battery() options include info, missing, descriptive, distributions, correlations and category_stats
- missing: Includes get_proportion_missing()
- distributions: Includes find_caps()
- categories: Includes get_sample_stats() and get_multi_group_stats()
- outcomes: Includes stats_by_outcome()
  - stats_by_outcome() analysis options include descriptives, proportions, regress. Note when the outcome type is continuous the descriptives option also includes a correlations analysis. 



## Notes

Currently the functions primarily rely on the use of pandas data frames. Numpy matrices can be passed in
however this may result in unexpected behavior. 



## Packages Required (see requirements.txt for versions)
- kneed
- matplotlib
- numpy
- pandas
- scikit-learn
- scikit-optimize
- scipy
- seaborn
- statsmodels
