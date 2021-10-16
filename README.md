# eagles

Creator: Jon-Frederick Landrigan



## Description:

This package contains utilities to perform tasks relating to data science including supervised and unsupervised machine learning, data exploration and statistical testing. 

For examples of how to use the package and its functions see the examples directory which contains jupyter notebooks pertaining to the main modules. 
## Install
To install you can use either 
```pip3 install eagles ``` to install from pypi   or 
```pip3 install git+https://github.com/JFLandrigan/eagles.git#"egg=eagles" ```  to install direct from the github repo in order to get the latest merges. Note when installing direct from github, while it may contain the latest updates, it may not be as stable as compared to releases installed from pypi. 

Once installed it can be imported like any other python package. For example:

```
from eagles.Supervised.tuner import SupervisedTuner
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
| "knn_clf" : KNeighborsClassifier      | "ridge":Ridge                            |
| "nn" : MLPClassifier                  | "elastic" : ElasticNet                   |
| "ada_clf" : AdaBoostClassifier        | "poisson":PoissonRegressor               |
| "et_clf": ExtraTreesClassifier        | "svr" : SVR                              |
| "vc_clf"  :VotingClassifier           | "knn_regress" : KNeighborsRegressor      |
|                                       | "ada_regress" : AdaBoostRegressor        |
|                                       | "et_regress": ExtraTreesRegressor        |
|                                       | "vc_regress" : VotingRegressor           |

Defaults:

VotingClassifier: Estimators - RandomForestClassifier and LogisticRegression, Voting - Hard, Weights - Uniform

VotingRegressor: Estimators - RandomForestRegressor and LinearRegression, Weights - Uniform

Note that sklearn pipeline objects can also be preset and passed in directly for evaluation. See the examples in examples/Supervised_Tuning.ipynb



## Metric Options

When setting up the SupervisedTuner() class the tune_metric argument is used for the parameter search and the *eval_metrics* argument is used for the final model evaluation (eval metrics should be passed in as a list). If no metrics are passed in (for tuning and/or eval) classification problems will default to 'f1' and regression problems will default to 'mse'. Note that for multi-class classification problems the metrics default to "macro" averages. 

| Classification                                               | Regression                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 'accuracy'                                                   | 'mse' - mean square error                                    |
| 'f1'                                                         | 'rmse' - root mean square error                              |
| 'precision'                                                  | 'mae' - mean absolute error                                  |
| 'recall'                                                     | 'mape' - mean absolute percent error (note zeros are excluded) |
| 'roc_auc' - Area Under the Receiver Operating Characteristic Curve | 'r2' - r squared                                             |
| 'precision_recall_auc' - Area Under the Precision Recall Curve |                                                              |



## SupervisedTuner() tuner options for hyperparameter searches

- 'random_cv' : sklearn implementation of random parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'bayes_cv' :  scikit-optimize implementation of bayes parameter search. Can pass in n_jobs to parallelize fits and n_iterations to determine total number of fits. 
- 'grid_cv' : sklearn implementation of paramter grid search. Can pass in n_jobs to parallelize fits.



## How to pass in parameters for hyperparameter tuning

- Single model: pass in relative parameters as a dictionary with key (parameter) value (listed parameter setting) pairs

- Model embedded within a pipeline: pass in relative parameters as dictionary  with key (parameter) value (listed parameter setting) pairs. Note that the parameter keys should follow the format  ```clf__< parameter >```

- Models embedded within a VotingClassifier or Voting Regressor: pass in relative parameters as dictionary  with key (parameter) value (listed parameter setting) pairs. Note that the parameter keys should follow the format (note the following example assumes a random forest and a logistic regression):

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



## Logging

The SupervisedTuner() class allows the user to log out a text file of the models performance as well as parameters tested, metrics, feature importance and more. The user can also log out the model and data used by passing in a list. See the following two examples:

```
# text log only 
tuner = SupervisedTuner(
    eval_metrics=["accuracy", "f1", "roc_auc"],
    num_cv=5,
    bins=None,
    num_top_fts=None,
    get_ft_imp=True,
    random_seed=4,
    binary=True,
    disp=True,
    log="log",
    log_name="model_eval_test.txt",
    log_path=None,
    log_note="This is a test of the model eval function",
)

# text, data and model
tuner = SupervisedTuner(
    tune_metric='f1',
    tuner="grid_cv",
    eval_metrics=["accuracy", "f1", "precision", "precision_recall_auc"],
    num_cv=5,
    bins=None,
    num_top_fts=None,
    get_ft_imp=True,
    random_seed=None,
    n_jobs=2,
    binary=True,
    disp=True,
    log=["log","mod","data"],
    log_name="model_tunetest_test.txt",
    log_path=None,
    log_note=note,
)

```

**Note that if no log path is passed in a data subdirectory will be created in eagles/eagles/Supervised/utils/**



### Unsupervised Model Natively Supported

Currently the functions primarily support the following the sklearn algorithms however other model objects can be passed in assuming they support the ```fit_predict()``` methodology like other sklearn clustering algorithms. 

- "kmeans"
- "agglomerativeclustering"
- "dbscan"

### Unsupervised Metrics

When ```unsupervised_tuner.find_optimal_clusters()``` with K-Means or Agglomerative Clustering is used the following metrics can be used to find the "optimal" or "suggested "number of clusters however thorough analysis should be performed.

- "max_sil" : After generating models based on the range of cluster numbers desired the algorithm will pick the optimal number of clusters as the number of  clusters which resulted in the highest max silhouette score. 
- "knee_wss"  :  After generating models based on the range of cluster numbers desired the ```KneeLocator()``` method used (provided by the kneed package) to find the "elbow" or point at which increasing the number of clusters does not significantly decrease the amount of within cluster variability. 

Note the DBSCAN algorithm uses internal methods to find the optimal number of clusters. 



## Exploratory Module

The Exploratory module contains functions and tools for performing exploratory data analysis on pandas data frames. Currently the module includes the following:

- explore: Includes run_battery(), get_base_descriptives() and get_correlations()
  - run_battery() options include info, missing, descriptive, distributions, correlations and category_stats
- missing: Includes get_proportion_missing()
- distributions: Includes find_caps()
- categories: Includes get_sample_stats() and get_multi_group_stats()
- outcomes: Includes stats_by_outcome()
  - stats_by_outcome() analysis options include descriptives, proportions, regress. Note when the outcome type is continuous the descriptives option also includes a correlations analysis. 



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
