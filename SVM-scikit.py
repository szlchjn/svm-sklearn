from sklearn import svm, datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=2)

cls = svm.SVC(kernel='rbf', C=1.0)

raw_model = cls.fit(X_train, y_train)

raw_train_score = raw_model.score(X_train, y_train)
raw_test_score = raw_model.score(X_test, y_test)

print('Train accuracy: {:.3f} %'.format(raw_train_score*100))
print('Test accuracy: {:.3f} %'.format(raw_test_score*100))

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

scaled_model = cls.fit(X_train_scaled, y_train)

scaled_train_score = scaled_model.score(X_train_scaled, y_train)
scaled_test_score = scaled_model.score(X_test_scaled, y_test)

print('\nScaled train accuracy: {:.3f} %'.format(scaled_train_score*100))
print('Scaled test accuracy: {:.3f} %'.format(scaled_test_score*100))


ax = plt.subplot(111)
ax.set_yticks([1, 1.8, 3, 3.8])
ax.set_yticklabels(['Scaled train', 'Scaled test', 'Raw train', 'Raw test'])
ax.set_xlim([0.5, 1])
ax.set_xlabel('Accuracy')
ax.barh(1, scaled_train_score, color='C1', align='center')
ax.barh(1.8, scaled_test_score, color='C', align='center')
ax.barh(3, raw_train_score, color='C1', align='center')
ax.barh(3.8, raw_test_score, color='C', align='center')

plt.show()