# eagles

Creator: Jon-Frederick Landrigan

## Description:
This repository contains utilities to perform tasks relating to data science 
including supervised and unsupervised machine learning, data exploration and statistical testing.
The functions primarily act as utility wrappers.

For examples of how to used the functions contained within the package see the following jupyter notebooks:
- Supervised Tuning.ipynb
- Unsupervised Tuning.ipynb

## Logging
The supervised_tuner model_eval() and tune_test_model() allow for logging of the models being tested.
Note that is no log path is passed in a data subdirectory will be created in eagles/eagles/Supervised/utils/

## How to use it?
Currently the package is only available via github. The simplest way to use the 
package in its current state is to download the repo and add it to the path you are working in.

## Notes
This package is still under heavy development and testing. 
Currently the functions primarily rely on the use of pandas data frames. Numpy matrices can be passed in
however this may result in unexpected behavior. 

## Packages Required (see requirements.txt for exact versions)
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