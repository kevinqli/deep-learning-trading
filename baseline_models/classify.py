from collections import defaultdict
import csv
import numpy as np
import pickle
from sklearn.linear_model import RidgeClassifier as RC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

DIR = '../data/'
train_prices_path = DIR + 'train_inputs.pkl'
train_deltas_path = DIR + 'train_labels.pkl'
val_prices_path = DIR + 'val_inputs.pkl'
val_deltas_path = DIR + 'val_labels.pkl'

NUM_FEATURES = 30
MODELS = ["RF", "RC", "GB"]

# Return 3-d np array of inputs and 1-d np array of labels
def get_inputs_and_labels():
    return pickle.load(open(train_prices_path, 'rb')), pickle.load(open(train_deltas_path, 'rb')), \
            pickle.load(open(val_prices_path, 'rb')), pickle.load(open(val_deltas_path, 'rb'))

# Return trained RF classifier
def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

# Return trained ridge classifier
def ridge_classifier(X_train, y_train):
    clf = RC()
    clf.fit(X_train, y_train)
    return clf

# Return trained gradient boosting classifier
def gradient_boosting_classifier(features, target):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    clf.fit(features, target)
    return clf

# Output training and validation accuracies
def train_model(X_train, y_train, X_val, y_val, model):
    if model == "RF":
        trained_model = random_forest_classifier(X_train, y_train)
        predictions = trained_model.predict(X_val)
        mean_train_acc = accuracy_score(y_train, trained_model.predict(X_train))
        mean_val_acc = accuracy_score(y_val, predictions)

    elif model == "RC":
        train_model = ridge_classifier(X_train, y_train)
        mean_train_acc = clf.score(X_train, y_train)
        mean_val_acc = clf.score(X_val, y_val)

    elif model == "GB":
        trained_model = gradient_boosting_classifier(X_train, y_train)
        predictions = trained_model.predict(X_val)
        mean_train_acc = accuracy_score(y_train, trained_model.predict(X_train))
        mean_val_acc = accuracy_score(y_val, predictions)

    else:
        print "Model not found"

    print 'Model: %s' % model
    print 'train acc: %f, val acc: %f' % (mean_train_acc, mean_val_acc)

def main():
    X_train, y_train, X_val, y_val = get_inputs_and_labels()
    print 'Predicting stock trends...'
    for model in models:
        train_model(X_train, y_train, X_val, y_val, model)

if __name__ == '__main__':
    main()



