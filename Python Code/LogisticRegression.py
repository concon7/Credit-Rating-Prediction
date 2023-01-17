import numpy as np
import matplotlib.pyplot as plt
import itertools
import pdb

def normalize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def augment_feature_vector(x):
    #  Adds the x[i][0] = 1 feature for each data point x[i].
    #  Args: x - a NumPy matrix of n data points, each with d - 1 features
    #  Returns: x augmented, an (n, d) NumPy array with the added feature for each datapoint
    column_of_ones = np.zeros([len(x), 1]) + 1
    return np.hstack((column_of_ones, x))

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    # Args: x - a NumPy matrix of n data points, each with d - 1 features
    return 1 / (1 + np.exp(-x))

def probability(weights, x):
    # Computes the weighted sum of inputs
    # Args: x - a NumPy matrix of n data points, each with d - 1 features
    # Returns the probability after passing through sigmoid
    return sigmoid(np.dot(x, weights))

class LR:
    # Initialize the class for Logistic Regression with Binary Cross Entropy Loss Function
    # with theta initialized to the all-zeros array. Here, theta is a k by d array
    # where row j represents the parameters of our model for label j = 0, 1, ..., k-1
    # ARGS:
    # X - (n, d) NumPy array (n data points, each with d-1 or d features)
    # Y - (n, ) NumPy array containing the labels (a number from 0 ... k-1) for each data point
    # max_iter stop gradient descent if convergence before max interations (scalar)
    # normal - X for each feature (boolean)
    # biases -if bias parameter used then augment X (boolean)
    # eta - the learning rate (scalar)

    def __init__(self, x, y, eta=0.001, max_iter=1000, normal=True, biases=True, verbose=False):
        # data preparation and parameter initialization
        if normal:                    #  normalize each feature (column) of X
            self.X = normalize(x)
        else:
            self.X = x
        if biases:                    #  add a column of ones to the beginning of X matrix
            self.X = augment_feature_vector(self.X)

        #self.k = len(np.unique(y))    #  choose unique y labels
        #hot = np.eye(self.k)          #  implement one-hot encoding
        #self.Y = np.array([])
        #for j in y:
        #    self.Y = np.append(self.Y, hot[int(j)], axis=0)
        #self.Y = self.Y.reshape((y.shape[0], self.k))

        self.k = y.shape[1]
        self.Y = y
        self.eta = eta
        self.biasFlag = biases        #  true if bias parameter included
        self.normalFlag = normal
        self.n = self.X.shape[0]      #  n observations
        self.d = self.X.shape[1]      #  d = # features + 1 for bias if biases == True
        self.max_iters = max_iter     #  maximum number of iterations
        self.n_iters = 0              #  iterations to convergence

        # Log of Training
        self.loss_function_progression = []
        self.total_cost = 0.0

        # initialize LR weights/thetas to 1.0
        self.theta = np.zeros((self.d, self.k))  # weights or theta parameters

        # fit the model to the data
        self.train(verbose)

    def loss_function(self):
        # Computes the Cross Entropy Loss Function for all the training samples
        total_loss = -(1 / self.n) * np.sum(self.Y * np.log(probability(self.theta, self.X)) +
                                            (1 - self.Y) * np.log(1 - probability(self.theta, self.X)))
        return total_loss

    def train(self, verbose=False):
        # Runs batch gradient descent for a specified number of iterations on a dataset
        # with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
        # where row j represents the parameters of our model for label j for
        # j = 0, 1, ..., k-1
        #
        # Returns:
        #    theta - (k, d) NumPy array that is the final value of parameters theta, k classifications / d/d-1 features
        #    cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
        i = 0
        converge = False
        loss = 0
        while ((not converge) and (i < self.max_iters)):
            prev_loss = loss
            loss = self.loss_function()
            self.loss_function_progression.append(loss)
            self.theta = self.theta - self.eta * self.gradient_descent_iteration()
            i += 1
            converge = np.abs(loss - prev_loss) < 1e-6
            if verbose:
                print("Iter: ",i, "LogLoss: ", loss, "\nThetas: \n")
                print(self.theta)
                print("Check if Probabilities sum = 1:\n")
                print(probability(self.theta, self.X).sum(axis=1))
        self.n_iters = i

    def gradient_descent_iteration(self):
        # Runs one step of gradient descent
        # Returns: the first derivative of the Cross Entropy Loss Function
        # theta - (k, d) NumPy array that is the final value of parameters theta
        # Computes the gradient of the cost function at the point theta
        return (1.0 / self.n) * np.dot(self.X.T, probability(self.theta, self.X) - self.Y)

    def predict_classification(self, x_test):
        # Makes predictions by classifying a given dataset
        # Args:
        #    x_test - (m, d) NumPy array (m test data points, each with d-1 or d features)
        # Returns:
        #    Y - (m, ) NumPy array, containing the predicted label (0 to k-1) for each data point
        if self.normalFlag:  # normalize each feature (column) of X
            x_test = normalize(x_test)
        if self.biasFlag:
            x_test = augment_feature_vector(x_test)
        probabilities = probability(self.theta, x_test)
        return np.argmax(probabilities, axis=1)

    def score(self, x_vars, y_labels):
        #
        #  Returns the % of correct prediction/total labels between y_labels and what classifier predicts
        #  Returns confusion matrix
        #
        # Args:
        #    X - (m, d) NumPy array (m datapoints each with d-1 or d features)
        #    Y - (m, ) NumPy array containing the labels (0, ..., k-1) for each data point
        #
        error_count = 0
        predict_labels = self.predict_classification(x_vars)
        cnf_matrix = np.zeros((self.k, self.k))
        for i in range(y_labels.shape[0]):
            cnf_matrix[int(y_labels[i]), int(predict_labels[i])] += 1
        return np.mean(predict_labels == y_labels), cnf_matrix


def plot_cost_function_over_time(cost_function_history, title="Loss Function History"):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.title(title)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')


def summaryLogReg(X, Y, k, N, class_labels, show=True, fname="plt", title="", equalNoLabels=False):

    #  Ensure each training dataset has equla number of number of labels

    if equalNoLabels:
        num_labels = Y.sum(axis=0)
        lowest_label = np.argmin(num_labels)
        num_classes = Y.shape[1]
        index_lowest_label = np.where(Y[:, lowest_label] > 0)[0]   # np.where returns a tuple
        n1 = len(index_lowest_label)
        randomOrder = np.random.permutation(n1)
        lls = int(8 * n1 / 10)  # lowest label number samples for training
        index_training = index_lowest_label[randomOrder[0:lls]]
        index_rest_labels = np.where(Y[:, lowest_label] == 0)[0]  # np.where returns a tuple
        n2 = len(index_rest_labels)
        randomOrder = np.random.permutation(n2)
        index_training = np.concatenate((index_training, index_rest_labels[randomOrder[0:int((num_classes-1) * lls)]]))
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

    # Multi-class logistic regression

    model = LR(x_train, y_train, eta=0.2, max_iter=2000, normal=True, biases=True, verbose=False)
    plot_cost_function_over_time(model.loss_function_progression,
                                 title="{0} Loss Function Convergence".format(title))
    show_or_save(showGraphs=show, name="{0}_loss".format(fname))

    score, confusion_matrix = model.score(x_train, np.argmax(y_train, axis=1))
    plot_confusion_matrix(confusion_matrix,  class_labels,
                          title="{0} Confusion Matrix\nTraining Data Score: {1:.3f}".format(title, score))
    show_or_save(showGraphs=show, name="{0}_CM_train".format(fname))

    y_pred = model.predict_classification(x_test)
    score, confusion_matrix = model.score(x_test, np.argmax(y_test, axis=1))
    plot_confusion_matrix(confusion_matrix, class_labels,
                          title="{0} Confusion Matrix\nTesting Data Score: {1:.3f}".format(title, score))

    show_or_save(showGraphs=show, name="{0}_CM_test".format(fname))


def show_or_save(showGraphs=True, name="plt"):
    if showGraphs:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


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