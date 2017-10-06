import featurizer


HAIR, FACE, BKG = 0, 1, 2

def basic(val=False):
    d = {
        'booster': 'gbtree',
        'num_class': 3,
        'objective': 'multi:softprob',
        'eta': 1.0,
        'gamma': 1.0,
        'min_child_weight': 1,
        'max_depth': 3,
        'num_round': 50,
        'save_period': 1,
        'eval_train': 1,
    }
    if not val: d['eval_metric'] = ['merror', 'mlogloss']
    return d


def feats1():
    window = 11
    fts = ['loc', 'stats', 'hog']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }


def feats2():
    window = 11
    fts = ['loc', 'col', 'stats', 'kpolar', 'kmeancol']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }

Conf = feats2()
