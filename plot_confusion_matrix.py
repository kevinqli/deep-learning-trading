import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class_names = ['no increase', 'increase']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() - 0.05
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        print cm[i, j]
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
train_cnf_matrix = np.array([[3252, 3113], [3193, 3606]]).T
eval_cnf_matrix = np.array([[27, 25], [3, 15]]).T
np.set_printoptions(precision=2)

# Set model for saving purposes
model = 'shallow_lstm'

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=class_names, normalize=True)
plt.savefig('confusion_matrixes/%s_train.png' % model)

plt.figure()
plot_confusion_matrix(eval_cnf_matrix, classes=class_names, normalize=True)
plt.savefig('confusion_matrixes/%s_val.png' % model)
