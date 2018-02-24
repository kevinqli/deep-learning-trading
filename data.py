from alpha_vantage.timeseries import TimeSeries
import csv
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

API_KEY = 'K4GGGZOT5MLPQ97T'

SYM_FILE = 'companylist.csv'
RAW_DATA_DIR = 'raw_data'
TRAIN_DATA_DIR = 'data'
WINDOW_SIZE = 60

TS = TimeSeries(key=API_KEY, output_format='pandas')


def get_syms():
    syms = []
    with open(SYM_FILE) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            syms.append(row[0])
    return syms


def load_sym(sym):
    with open(os.path.join(RAW_DATA_DIR, sym)) as f:
        return pickle.load(f)


def split(prices, window=WINDOW_SIZE):
    n = len(prices)
    indices = np.arange(window, n, window)
    inputs = np.split(prices[:indices[-2]], indices[:-2])
    deltas = (prices[indices[1:]] - prices[indices[:-1]]) /\
        prices[indices[:-1]]
    assert len(inputs) == len(deltas)
    return inputs, deltas


def split_all_data_and_save(window=WINDOW_SIZE, label='4. close'):
    all_inputs = []
    all_deltas = []
    for sym in os.listdir(RAW_DATA_DIR):
        if sym[0] != '.':
            data = load_sym(sym)
            if 'data' in data:
                prices = data['data'][label].as_matrix()
                if len(prices) > 2 * window:
                    print 'Loading data from symbol %s...' % sym
                    inputs, deltas = split(prices, window=window)
                    all_inputs.extend(inputs)
                    all_deltas.append(deltas)
    all_inputs = np.concatenate(all_inputs).reshape(-1, window)
    all_deltas = np.concatenate(all_deltas).reshape(-1, 1)
    means = np.mean(all_inputs, axis=1, keepdims=True)
    stds = np.std(all_inputs, axis=1, keepdims=True)
    all_inputs = (all_inputs - means) / stds
    with open(os.path.join(TRAIN_DATA_DIR, 'prices'), 'w') as f:
        pickle.dump(all_inputs, f)
    with open(os.path.join(TRAIN_DATA_DIR, 'deltas'), 'w') as f:
        pickle.dump(all_deltas, f)
    return all_inputs, all_deltas


def output_to_txt(array, path):
    with open(path, 'w') as f:
        for arr in array:
            f.write(' '.join(map(str, arr)) + '\n')

def split_train_dev_test_to_file():
    with open(os.path.join(TRAIN_DATA_DIR, 'prices')) as f:
        prices = pickle.load(f)
    with open(os.path.join(TRAIN_DATA_DIR, 'deltas')) as f:
        deltas = pickle.load(f)
    n = prices.shape[0]
    print prices.shape
    print deltas.shape
    indices = np.random.permutation(n)
    train_idx, eval_idx, test_idx = \
        indices[:int(0.9*n)], indices[int(0.9*n):int(0.95*n)], \
        indices[int(0.95*n):]
    output_to_txt(prices[train_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'train_prices.txt'))
    output_to_txt(prices[eval_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'eval_prices.txt'))
    output_to_txt(prices[test_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'test_prices.txt'))
    output_to_txt(deltas[train_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'train_deltas.txt'))
    output_to_txt(deltas[eval_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'eval_deltas.txt'))
    output_to_txt(deltas[test_idx, :],
                  os.path.join(TRAIN_DATA_DIR, 'test_deltas.txt'))


def get_sym_and_save(sym, interval='1min'):
    if os.path.exists(os.path.join(RAW_DATA_DIR, sym)):
        print 'Symbol %s already downloaded. Skipping...' % sym
        return
    try:
        data, metadata = TS.get_intraday(symbol=sym, interval=interval,
                                         outputsize='full')
    except ValueError:
        print 'Symbol %s does not exist! Skipping...' % sym
        with open(os.path.join(RAW_DATA_DIR, sym), 'w') as f:
            pickle.dump({}, f)
        return
    with open(os.path.join(RAW_DATA_DIR, sym), 'w') as f:
        pickle.dump({
            'data': data,
            'metadata': metadata
        }, f)


def get_all_raw_data(interval='1min'):
    syms = get_syms()
    for i, sym in enumerate(syms):
        print '(%d/%d) Getting data for %s...' % (i, len(syms), sym)
        get_sym_and_save(sym)


def main():
    get_all_raw_data()
    split_all_data_and_save()


if __name__ == '__main__':
    main()
