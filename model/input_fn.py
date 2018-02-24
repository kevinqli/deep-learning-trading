"""Create the input data pipeline using `tf.data`"""

import numpy as np
import tensorflow as tf

def load_prices_and_deltas(prices_file, deltas_file, params):
	"""Create tf.data instance from txt files

	Args:
		prices_file: (string) file containing all sequences of prices
		deltas_file: (string) file containing corresponding deltas
		params: data parameters

	Returns:
		prices, deltas: (tf.Dataset) yielding list of stock prices, lengths, and deltas
	"""

	prices, deltas = [], []

	with open(prices_file, 'r') as pf:
		for line in pf:
			prices.append([[float(num)] for num in line.strip().split(' ')])

	with open(deltas_file, 'r') as df:
		for line in df:
			deltas.append(float(line.strip()))

	prices = tf.data.Dataset.from_tensor_slices(tf.constant(prices, dtype=tf.float32))
	deltas = tf.data.Dataset.from_tensor_slices(tf.constant(deltas, dtype=tf.float32))
	return prices, deltas

def input_fn(mode, prices, deltas, params):
    """Input function

    Args:
        mode: (bool) 'train', 'eval'
                     At training, we shuffle the data and have multiple epochs
        prices: (tf.Dataset) yielding list of historical prices
        deltas: (tf.Dataset) yielding corresponding list of deltas
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

     # Load all the dataset in memory for shuffling if training
    is_training = (mode == 'train')
    buffer_size = params.train_size if is_training else 1

    # Zip the prices and the deltas together
    dataset = tf.data.Dataset.zip((prices, deltas))

    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (prices, deltas) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'prices': prices,
        'deltas': deltas,
        'iterator_init_op': init_op
    }

    return inputs
