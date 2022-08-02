# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistic Regression (custom parameters: max_iter=1e5) and a Random Forest Classifier (custom parameters: n_estimators=20, max_depth=10) were trained. Based on the better f_beta score the Random Forest Classifier was selected.

* Model version: 1.0.0
* Model date: 01 ago 2022

## Intended Use
The model can be used for predicting income classes on census data. There are two income classes >50K and <=50K (binary classification task).

## Training Data
The UCI Census Income Data Set was used for training. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
For training 80% of the 32561 rows were used (26561 instances) in the training set.
## Evaluation Data
For evaluation 20% of the 32561 rows were used (6513 instances) in the test set.
## Metrics
Three metrics were used for model evaluation (performance on test set):
* precision: 0.7833
* recall: 0.5092
* fbeta: 0.6172

## Ethical Considerations
Since the dataset consists of public available data with highly aggregated census data no harmful unintended use of the data has to be adressed.
## Caveats and Recommendations
It would be meaningful to perform an hyperparameter optimization to improve the model performance.