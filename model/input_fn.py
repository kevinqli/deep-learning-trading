"""Create the input data pipeline using `tf.data`"""

import numpy as np
import pickle
import tensorflow as tf

NUM_FEATURES = 30

def load_prices_and_deltas(inputs_file, labels_file, params):
    """Create tf.data instance from txt files
    
    Args:
        prices_file: (string) file containing all sequences of prices
        deltas_file: (string) file containing corresponding deltas
        params: data parameters

    Returns:
        prices, deltas: (tf.Dataset) yielding list of stock prices, lengths, and deltas
    """
    inputs = pickle.load(open(inputs_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))
    inputs = tf.data.Dataset.from_tensor_slices(tf.constant(inputs, dtype=tf.float32))
    labels = tf.data.Dataset.from_tensor_slices(tf.constant(labels, dtype=tf.float32))
    return inputs, labels

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

    seed = tf.placeholder(tf.int64, shape=())
    dataset = (dataset
        .shuffle(buffer_size=buffer_size, seed=seed)
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
        'iterator_init_op': init_op,
        'seed': seed
    }

    return inputs
