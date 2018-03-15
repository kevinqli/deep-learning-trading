import cPickle as pickle
import numpy as np
import os

TRAIN_DATA_DIR = 'data'
WINDOW_SIZE = 60

def generate_sin_and_delta(window=WINDOW_SIZE, offset_x=0, offset_y=0, scale=1,
                 noise_std=0.1):
    def f(x):
        return scale * np.sin(x - offset_x) + offset_y + \
            np.random.normal(0, noise_std)
    f1 = f(window)
    f2 = f(2 * window)
    delta = (f2 - f1) / f1
    return np.vectorize(f)(np.arange(window)), delta

def batch_generate():
    inputss = []
    deltas = []
    for _ in range(1000):
        inputs, delta = generate_sin_and_delta(offset_x=np.random.normal(0, 10),
                                               offset_y=np.random.normal(100, 50),
                                               scale=np.random.normal(10, 5),
                                               noise_std=0.1)
        inputss.append(inputs)
        deltas.append(delta)
    inputss = np.array(inputss)
    deltas = np.array(deltas).reshape(-1, 1)
    with open(os.path.join(TRAIN_DATA_DIR, 'prices'), 'w') as f:
        pickle.dump(inputss, f)
    with open(os.path.join(TRAIN_DATA_DIR, 'deltas'), 'w') as f:
        pickle.dump(deltas, f)
