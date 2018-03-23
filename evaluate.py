"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn, load_prices_and_deltas
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Get paths for dataset
    path_test_prices = os.path.join(args.data_dir, 'test_inputs.pkl')
    path_test_deltas = os.path.join(args.data_dir, 'test_labels.pkl')

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_prices, test_deltas = load_prices_and_deltas(path_test_prices, path_test_deltas, params)

    # Create the two iterators over the two datasets
    test_inputs = input_fn('test', test_prices, test_deltas, params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('test', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
