import numpy as np
import logging


def score_fn(y_preds, y_true, to_print=True):
    accuracy = np.sum(y_preds == y_true) / len(y_preds)

    # correct prediction (true positives)
    t = np.sum(y_preds == y_true)
    f = len(y_preds) - t

    if to_print:
        print('Model accuracy', accuracy)
        print('Correct prediction ', t)
        print('Incorrect prediction ', f)

    logging.info(f'Model accuracy : {accuracy} ')
    logging.info(f'Correct prediction : {t}')
    logging.info(f'Incorrect prediction : {f}')
    logging.info(f'Model accuracy : {accuracy}')
