from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from time import time
from read_data import load_mnist

X, y = load_mnist('data')
scaler = MinMaxScaler()
scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X[::10], y[::10], test_size=1000)


def svm(c, kernel='poly', degree=2):
    if kernel == 'poly':
        svm_model = SVC(kernel=kernel, degree=degree, C=c)
    else:
        svm_model = SVC(kernel=kernel, C=c)
    svm_model.fit(X_train, y_train)
    return svm_model


if __name__ == "__main__":
    kernels = ['rbf']
    c = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    # gamma = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    # kernels = ['linear']
    for k in kernels:
        for cc in c:
            print('%s(C=%.1f):' % (k, cc))
            # train model
            print('training...')
            start = time()
            model = svm(cc, kernel=k)
            print('training finished in %.2fs' % (time() - start))

            # print('acc: %.2f' % model.score(X_test, y_test))

            # predict
            y_pred = model.predict(X_test)
            # print(len(set(y_pred))-len(set(y_test)))
            print(classification_report(y_test, y_pred, digits=4))

