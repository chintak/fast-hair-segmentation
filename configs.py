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


def subsample(val=False):
    d = {
        'booster': 'gbtree',
        'num_class': 3,
        'objective': 'multi:softprob',
        'subsample': 0.5,
        'gamma': 1.0,
        'max_depth': 5,
        'num_round': 50,
    }
    if not val: d['eval_metric'] = ['merror', 'mlogloss']
    return d


def colsubsample(val=False):
    d = {
        'booster': 'gbtree',
        'num_class': 3,
        'objective': 'multi:softprob',
        'colsubsample_bytree': 0.5,
        'gamma': 1.0,
        'max_depth': 5,
        'num_round': 50,
    }
    if not val: d['eval_metric'] = ['merror', 'mlogloss']
    return d


def subcolsub(val=False):
    d = {
        'booster': 'gbtree',
        'num_class': 3,
        'objective': 'multi:softprob',
        'subsample': 0.5,
        'colsubsample_bysplit': 0.3,
        'gamma': 1.0,
        'max_depth': 5,
        'num_round': 50,
    }
    if not val: d['eval_metric'] = ['merror', 'mlogloss']
    return d


def imbalance(val=False):
    d = {
        'booster': 'gbtree',
        'num_class': 3,
        'objective': 'multi:softprob',
        'max_delta_step': 1.0,
        'colsubsample_bytree': 0.75,
        'gamma': 1.0,
        'max_depth': 5,
        'num_round': 50,
    }
    if not val: d['eval_metric'] = ['merror', 'mlogloss']
    return d


def feats0():
    window = 11
    fts = ['loc', 'stats', 'hog']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }


def feats1():
    window = 11
    fts = ['loc', 'col', 'stats', 'hog']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }


def feats2():
    window = 7
    fts = ['loc', 'col', 'stats', 'kpolar', 'kmeancol']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }


def feats3():
    window = 11
    fts = ['loc', 'col', 'stats', 'kpolar']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }


def feats4():
    window = 11
    fts = ['loc', 'col', 'stats', 'kpolar', 'kmeancol']
    return {
        'WINDOW': window,
        'FEATS': featurizer.Featurizer(types=fts, window=window),
    }
