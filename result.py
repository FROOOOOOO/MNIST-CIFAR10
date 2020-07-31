import matplotlib.pyplot as plt


def show_knn():
    input_values = [1, 3, 5, 7, 9, 11, 13, 15]
    pre = [0.9691, 0.9709, 0.9693, 0.9699, 0.9667, 0.9676, 0.9662, 0.9642]
    recall = [0.9688, 0.9701, 0.9685, 0.9691, 0.9657, 0.9665, 0.9651, 0.9630]
    f1 = [0.9689, 0.9704, 0.9687, 0.9694, 0.9659, 0.9669, 0.9654, 0.9634]
    plt.plot(input_values, pre, 'r', label='precision')
    plt.plot(input_values, recall, 'g', label='recall')
    plt.plot(input_values, f1, 'b', label='f1-score')
    plt.legend()
    plt.xlabel('k')
    plt.xticks(input_values)
    plt.show()


def show_svm():
    names = ['rbf', 'sigmoid', 'poly', 'linear']
    pre = [0.9792, 0.7743, 0.9771, 0.8953]
    recall = [0.9791, 0.7725, 0.9769, 0.8697]
    f1 = [0.9791, 0.7723, 0.9770, 0.8682]
    x = list(range(len(names)))
    total_w, n = 0.8, 3
    width = total_w/n
    plt.bar(x, pre, width=width, label='precision', fc='r')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, recall, width=width, label='recall', tick_label=names, fc='g')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, f1, width=width, label='f1-score', fc='b')
    plt.legend()
    plt.xlabel('kernel function')
    # plt.xticks(names)
    plt.show()
    # name_list = ['A', 'B', 'C', 'D']
    # num_list = [10, 15, 16, 28]
    # num_list2 = [10, 12, 18, 26]
    # x = list(range(len(num_list)))
    # total_width, n = 0.8, 2
    # width = total_width / n
    # plt.bar(x, num_list, width=width, label='1', fc='b')
    # for i in range(len(x)):
    #     x[i] += width
    # plt.bar(x, num_list2, width=width, label='2', tick_label=name_list, fc='g')
    # plt.legend()
    # plt.show()


def show_poly():
    Cs = [0.1, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0]
    acc = [0.855, 0.926, 0.936, 0.939, 0.937, 0.938, 0.938, 0.938, 0.938]
    plt.plot(Cs, acc, 'r', label='accuracy')
    plt.legend()
    plt.xlabel('C')
    plt.xticks([0.1, 5.0, 15.0, 30.0, 50.0, 100.0])
    plt.show()


def show_rbf():
    Cs = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    acc = [0.9531, 0.9751, 0.9791, 0.9804, 0.9811, 0.9810, 0.9810, 0.9810]
    plt.plot(Cs, acc, 'r', label='accuracy')
    plt.legend()
    plt.xlabel('C')
    plt.xticks([0.1, 5.0, 10.0, 20.0, 50.0, 100.0])
    plt.show()


if __name__ == '__main__':
    show_rbf()
