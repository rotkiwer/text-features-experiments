import pickle
from cloud.serialization.cloudpickle import dump


def read_selector(name):
    with open(name + '.selector', 'rb') as f:
        name_saved, sel_train_time, selector = pickle.load(f)
    return sel_train_time


name = '20ng_t2_chi2+lda-'
for x in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    result = read_selector(name + str(x))
    print result
