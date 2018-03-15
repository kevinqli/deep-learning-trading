import numpy as np
import os
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

PATH = '../data'
REGULARIZER = 0.1

# Parses an individual file based on whether it contains prices or deltas
def parse_file(file_path, file_type):
	result = []
	with open(file_path, 'r') as f:
		if (file_type == 'prices'):
			for row in f:
				result.append([float(num) for num in row.strip().split(' ')])
		elif (file_type == 'deltas'):
			for row in f:
				result.append(float(row.strip()))
		else:
			raise ValueError('Unknown file type: %s' % file_type)
	return np.array(result)

# Returns training and evaluation data
def get_data():
	train_prices_path = os.path.join(PATH, 'train_prices.txt')
	train_deltas_path = os.path.join(PATH, 'train_deltas.txt')
	eval_prices_path = os.path.join(PATH, 'eval_prices.txt')
	eval_deltas_path = os.path.join(PATH, 'eval_deltas.txt')

	X_train = parse_file(train_prices_path, 'prices')
	y_train = parse_file(train_deltas_path, 'deltas')
	X_eval = parse_file(eval_prices_path, 'prices')
	y_eval = parse_file(eval_deltas_path, 'deltas')

	return X_train, y_train, X_eval, y_eval

# Custom loss function: use sigmoid first to clamp predictions b/t 0 and 1
def custom_loss(truth, preds):
	sigmoid_preds = 1 / (1 + np.exp(-preds))
	return np.mean(-(np.multiply(truth, sigmoid_preds) + REGULARIZER*np.log(1 - sigmoid_preds)))

# Calculate geometric average of profit made from evaluation actions
def profit(y_eval, preds):
	sigmoid_preds = 1 / (1 + np.exp(-preds))
	return np.prod(1 + np.multiply(sigmoid_preds, y_eval)) ** (1.0 / len(preds))

# Linear regression using SGD 
def main():
	X_train, y_train, X_eval, y_eval = get_data()
	loss_scorer = make_scorer(custom_loss, greater_is_better=False)
        clf = GridSearchCV(linear_model.SGDRegressor(), param_grid={'n_iter':[1000]}, scoring=loss_scorer)   
        
        # Fit training examples
	clf.fit(X_train, y_train)
	# Make predictions
	preds = clf.predict(X_eval)
	# Calculate evaluation statistics
	eval_loss = custom_loss(y_eval, preds)
	print('Eval loss: %f' % eval_loss)
	eval_profit = profit(y_eval, preds)
	print('Eval profit: %f' % eval_profit)

if __name__ == '__main__':
	main()
