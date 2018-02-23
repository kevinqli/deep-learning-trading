from alpha_vantage.timeseries import TimeSeries
import csv
import cPickle as pickle
import matplotlib.pyplot as plt
import os
from pprint import pprint

API_KEY = 'K4GGGZOT5MLPQ97T'

SYM_FILE = 'companylist.csv'
DATA_DIR = 'data'

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
    with open(os.path.join(DATA_DIR, sym)) as f:
        return pickle.load(f)


def get_sym_and_save(sym, interval='1min'):
    if os.path.exists(os.path.join(DATA_DIR, sym)):
        print 'Symbol %s already downloaded. Skipping...' % sym
        return
    try:
        data, metadata = TS.get_intraday(symbol=sym, interval=interval,
                                     outputsize='full')
    except ValueError:
        print 'Symbol %s does not exist! Skipping...' % sym
        with open(os.path.join(DATA_DIR, sym), 'w') as f:
            pickle.dump({}, f)
        return
    with open(os.path.join(DATA_DIR, sym), 'w') as f:
        pickle.dump({
            'data': data,
            'metadata': metadata
        }, f)


def get_all_data(interval='1min'):
    syms = get_syms()
    for i, sym in enumerate(syms):
        print '(%d/%d) Getting data for %s...' % (i, len(syms), sym)
        get_sym_and_save(sym)


def main():
    get_all_data()


if __name__ == '__main__':
    main()
