import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ReLU
import tarfile
import pdb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          x_label='Predicted label',
                          y_label='True label',
                          cmap=plt.cm.Blues,
                          anot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Confusion Matrix {0}:\n'.format(cm.shape), cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    if anot:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()


def plot_filters(conv, shape, title):
    plt.figure(figsize=(5, 4))
    cnt = 0
    m, n = shape
    for i in range(m):
        for j in range(n):
            cnt += 1
            plt.subplot(m, n, cnt)
            plt.imshow(conv[0, :, :, cnt])
            plt.axis('off')
    fig = plt.gcf()
    fig.suptitle(title)


def my_one_hot(targets):
    classes = np.unique(targets).shape[0]
    res = np.eye(classes)[np.array(targets).reshape(-1)]
    return res #.reshape(list(targets.shape)+[classes])


def show_or_save(showGraphs=True, name="plt"):
    if showGraphs:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


def plot_cost_function_over_time(loss_history, val_loss_history, title="Loss Function History"):
    plt.plot(range(len(loss_history)), loss_history, 'b', val_loss_history, 'r')
    plt.title(title)
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch #')


def summaryNNClass(X, Y, k, N, class_labels, show=True, fname="plt", title="",
                   batch_size=164, epochs=200, lr=0.001, reg='l2', equalNoLabels=False):

    #  Ensure each training dataset has equla number of number of labels

    if equalNoLabels:
        num_labels = Y.sum(axis=0)
        lowest_label = np.argmin(num_labels)
        num_classes = Y.shape[1]
        index_lowest_label = np.where(Y[:, lowest_label] > 0)[0]  # np.where returns a tuple
        n1 = len(index_lowest_label)
        randomOrder = np.random.permutation(n1)
        lls = int(8 * n1 / 10)  # lowest label number samples for training
        index_training = index_lowest_label[randomOrder[0:lls]]
        index_rest_labels = np.where(Y[:, lowest_label] == 0)[0]  # np.where returns a tuple
        n2 = len(index_rest_labels)
        randomOrder = np.random.permutation(n2)
        index_training = np.concatenate(
            (index_training, index_rest_labels[randomOrder[0:int((num_classes - 1) * lls)]]))
        index_testing = np.delete(np.arange(N), index_training)
        print("Training Portion {0:.2f} : {1} samples".format(len(index_training) / (len(index_training)
                                                            + len(index_testing)), len(index_training)))
    else:
        randomOrder = np.random.permutation(N)  # random sampling of observations
        s = int(8 * N / 10)  # split training vs. testing 80:20
        index_training = randomOrder[0:s]
        index_testing = randomOrder[s:N]

    x_train = X[index_training, :]
    y_train = Y[index_training, :]
    x_test = X[index_testing, :]
    y_test = Y[index_testing, :]

    num_classes = np.unique(y_train).shape[0]

    model = Sequential()
    model.add(Dense(16, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(Dense(8, activation='tanh'))
    model.add(Dense(k, activation='softmax', kernel_regularizer=reg))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    print(x_train.shape, y_train.shape)

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot loss & confusion matrix for test data
    plot_cost_function_over_time(hist.history['loss'], hist.history['val_loss'],
                                 title="{0} Neural Net Loss Function History".format(title))
    show_or_save(showGraphs=show, name="{0}_loss_NNClass".format(fname))

    # Plot the confusion matrix

    outputs = model.predict(x_test)
    pred_class = np.argmax(outputs, 1)
    true_class = np.argmax(y_test, 1)
    M = confusion_matrix(true_class, pred_class)
    plot_confusion_matrix(M, class_labels,
                          title="{0} Confusion Matrix - Testing Data\n Score: {1:.3f}".format(title, score[1]))
    show_or_save(showGraphs=show, name="{0}_CM_train".format(fname))
