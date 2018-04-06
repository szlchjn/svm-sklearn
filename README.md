# svm-sklearn
Support vector machine classifier with data standardization example in scikit
## Training SVM classifier on UCI ML Breast Cancer Wisconsin dataset.
### Effects of data standardization
First training was done on raw data, and the results show overfitting (100% on training data with test accuracy close
to random guessing ~60%)
###
Second training was done with the same hyperparameters (kernel='rbf', C=1.0), only with data stardardized by scikit's built-in preprocessing module. 
Rescaled data has [zero mean and unit variance](http://scikit-learn.org/stable/modules/preprocessing.html).
```python
from sklearn import preprocessing

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
```
###
With rescaled data SVM model learns to generalize well with both, training and testing, accuracies close to 98%
![figure_1](https://user-images.githubusercontent.com/30974121/37826036-5762b4dc-2e92-11e8-9aa1-8c93cadcbb0d.png)
