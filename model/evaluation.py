"""Tensorflow utility functions for evaluation"""

import logging
import numpy as np
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json


def evaluate_sess(sess, model_spec, num_steps, epoch, model_dir, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    profit = model_spec['profit']
    predictions = model_spec['predictions']
    labels = model_spec['labels']
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'], feed_dict={model_spec['seed']: epoch})
    sess.run(model_spec['metrics_init_op'])

    # Generate confusion matrix
    conf_matrix = np.zeros((2, 2), dtype=np.int32)

    # compute metrics over the dataset
    #total_profit = 1.0
    all_preds = []
    for _ in range(num_steps):
        _, profit_val, preds, label_vals = sess.run([update_metrics, profit, predictions, labels], feed_dict={model_spec['is_training']: True})
        #total_profit *= (1 + profit_val)
        for j in range(len(preds)):
            conf_matrix[preds[j][0]][int(label_vals[j])] += 1
        all_preds.append(preds)

    eval_preds_file = os.path.join(model_dir, 'eval_preds.txt')
    with open(eval_preds_file, 'w') as epf:
        for batch in all_preds:
            for pred in batch:
                epf.write(str(pred) + '\n')

    # Get geometric average of profit
    #avg_profit = total_profit ** (1 / num_steps)
    profit_string = "profit: " + str('ignore')

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.6f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string + " ; " + profit_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val, conf_matrix


def evaluate(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        metrics, _ = evaluate_sess(sess, model_spec, num_steps, model_dir)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)
