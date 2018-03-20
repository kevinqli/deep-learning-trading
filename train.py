"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.input_fn import input_fn, load_prices_and_deltas
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Get paths for dataset
    path_train_prices = os.path.join(args.data_dir, 'train_inputs.pkl')
    path_train_deltas = os.path.join(args.data_dir, 'train_labels.pkl')
    path_eval_prices = os.path.join(args.data_dir, 'eval_inputs.pkl')
    path_eval_deltas = os.path.join(args.data_dir, 'eval_labels.pkl')

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_prices, train_deltas = load_prices_and_deltas(path_train_prices, path_train_deltas, params)
    eval_prices, eval_deltas = load_prices_and_deltas(path_eval_prices, path_eval_deltas, params)

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_prices, train_deltas, params)
    eval_inputs = input_fn('eval', eval_prices, eval_deltas, params)
    logging.info("- done.")

    # Define the models
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
