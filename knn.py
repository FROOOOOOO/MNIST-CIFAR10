import read_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from time import time
import matplotlib.pyplot as plt
import random


def knn(k_size):
    if k_size > 0 and k_size % 2 == 1:
        model = KNeighborsClassifier(n_neighbors=k_size)
        model.fit(read_data.X_train, read_data.y_train)
        return model


if __name__ == '__main__':
    k_sizes = [1, 3, 5, 7, 9, 11, 13, 15]
    # k_sizes = [3]
    X_test, y_test = read_data.load_mnist('data', kind='t10k')
    for k_size in k_sizes:
        print('%d-NN:' % k_size)

        # train
        print('training...')
        start = time()
        model = knn(k_size)
        print('training finished in %.2f s' % (time() - start))

        # predict
        y_pred = []
        wrongs = []  # [[index, wrong_pred], ...]
        print('predicting...')
        start = time()
        for i, x_test in enumerate(X_test):
            pred = model.predict([x_test])[0]
            y_pred.append(pred)
            if y_test[i] != pred:
                wrongs.append([i, pred])
        print('prediction finished in %.2f s' % (time() - start))

        # analyze
        print(classification_report(y_test, y_pred, digits=4))

        # plt
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all', )
        ax = ax.flatten()
        sample = random.sample(wrongs, 25)

        for i in range(25):
            img = X_test[sample[i][0]].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('real: %d\npred: %d' %(y_test[sample[i][0]], sample[i][1]))

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
