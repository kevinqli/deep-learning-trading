from alpha_vantage.timeseries import TimeSeries
import csv
import pickle
import numpy as np
import os
import ta


def MA5(df):
    return ta.MA(df, 5)


def MA10(df):
    return ta.MA(df, 10)


def EMA20(df):
    return ta.EMA(df, 20)


def ROC(df):
    return ta.ROC(df, 5)


def ATR(df):
    return ta.ATR(df, 10)


def BBANDS(df):
    return ta.BBANDS(df, 20)


def ADX(df):
    return ta.ADX(df, 10, 5)


def MACD(df):
    return ta.MACD(df, 5, 20)


def RSI(df):
    return ta.RSI(df, 10)


def MFI(df):
    return ta.MFI(df, 10)


def CCI(df):
    return ta.CCI(df, 10)


def KELCH(df):
    return ta.KELCH(df, 20)


indicators = [MA5, MA10, EMA20, ROC, ATR, BBANDS, ta.PPSR, ta.STOK,
              ADX, MACD, RSI, MFI, CCI, KELCH]

NB_FEATURES = 30

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


def split(inputs, window=WINDOW_SIZE):
    n = len(inputs)
    indices = np.arange(window, n, window)
    split_inputs = np.split(inputs[:indices[-2]], indices[:-2])
    labels = inputs['Low'].ix[indices[1:]].as_matrix()\
        > inputs['High'].ix[indices[:-1]].as_matrix()
    assert len(split_inputs) == len(labels)
    return split_inputs, labels


def preprocess(data):
    data = data.rename(index=str, columns={x: x[3:].capitalize() for x in
                                           data.columns})
    data['Volume'] = np.log(1 + data['Volume'])
    data.index = range(len(data.index))
    for indicator in indicators:
        data = indicator(data)
    data = data.shift(-WINDOW_SIZE)[:-WINDOW_SIZE]
    return data


def split_all_data_and_save(window=WINDOW_SIZE, nb_features=NB_FEATURES):
    all_inputs = []
    all_labels = []
    for sym in os.listdir(RAW_DATA_DIR):
        if sym[0] != '.':
            data = load_sym(sym)
            if 'data' in data:
                inputs = preprocess(data['data'])
                if inputs.shape[0] > 2 * window:
                    print('Loading data from symbol %s...' % sym)
                    inputs, labels = split(inputs, window=window)
                    all_inputs.extend(inputs)
                    all_labels.append(labels)
    all_inputs = np.concatenate(all_inputs).reshape(-1, window, nb_features)
    all_labels = np.concatenate(all_labels).reshape(-1, 1)
    print(all_inputs.shape)
    print(all_labels.shape)
    assert all_inputs.shape[0] == all_labels.shape[0]
    with open(os.path.join(TRAIN_DATA_DIR, 'all_inputs.pkl'), 'wb') as f:
        pickle.dump(all_inputs, f)
    with open(os.path.join(TRAIN_DATA_DIR, 'all_labels.pkl'), 'wb') as f:
        pickle.dump(all_labels, f)
    return all_inputs, all_labels


def split_train_dev_test_to_file(inputs=None, labels=None):
    if inputs is None:
        with open(os.path.join(TRAIN_DATA_DIR, 'all_inputs.pkl'), 'rb') as f:
            inputs = pickle.load(f)
    if labels is None:
        with open(os.path.join(TRAIN_DATA_DIR, 'all_labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
    n = inputs.shape[0]
    print(inputs.shape)
    print(labels.shape)
    indices = np.random.permutation(n)
    train_idx, eval_idx, test_idx = \
        indices[:int(0.9*n)], indices[int(0.9*n):int(0.95*n)], \
        indices[int(0.95*n):]
    pickle.dump(inputs[train_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'train_inputs.pkl'), 'wb'))
    pickle.dump(inputs[eval_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'eval_inputs.pkl'), 'wb'))
    pickle.dump(inputs[test_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'test_inputs.pkl'), 'wb'))
    pickle.dump(labels[train_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'train_labels.pkl'), 'wb'))
    pickle.dump(labels[eval_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'eval_labels.pkl'), 'wb'))
    pickle.dump(labels[test_idx, :],
                open(os.path.join(TRAIN_DATA_DIR, 'test_labels.pkl'), 'wb'))


def get_sym_and_save(sym):
    if os.path.exists(os.path.join(RAW_DATA_DIR, sym)):
        print('Symbol %s already downloaded. Skipping...' % sym)
        return
    try:
        data, metadata = TS.get_daily(symbol=sym, outputsize='full')
    except ValueError:
        print('Symbol %s does not exist! Skipping...' % sym)
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
        print('(%d/%d) Getting data for %s...' % (i, len(syms), sym))
        get_sym_and_save(sym)


def main():
    if not os.path.exists(os.path.join(TRAIN_DATA_DIR, 'all_inputs.pkl')) \
            or not os.path.exists(os.path.join(TRAIN_DATA_DIR,
                                               'all_labels.pkl')):
        get_all_raw_data()
        split_all_data_and_save()
    split_train_dev_test_to_file()


if __name__ == '__main__':
    main()
