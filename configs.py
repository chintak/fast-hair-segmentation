import featurizer


HAIR, FACE, BKG = 0, 1, 2


def model_decorator(model_config_func):
    def wrapper(val=False):
        d = {}
        d['booster'] = 'gbtree'
        d['num_class'] = 3
        d['num_round'] = 50
        d['objective'] = 'multi:softprob'
        d['save_period'] = 1
        d['eval_train'] = 1
        d.update(model_config_func())
        if not val: d['eval_metric'] = ['merror', 'mlogloss']
        return d
    return wrapper


def feat_decorator(feat_func):
    """ Wrapper for feat_func. window, feats_list = feat_func() """
    def wrapper():
        window, fts = feat_func()
        return {
            'WINDOW': window,
            'FEATS': featurizer.Featurizer(types=fts, window=window),
        }
    return wrapper


@model_decorator
def basic():
    return {
        'eta': 1.0,
        'gamma': 1.0,
        'min_child_weight': 1,
        'max_depth': 3,
    }


@model_decorator
def small():
    return {
        'colsubsample_bytree': 0.5,
        'eta': 1.0,
        'gamma': 1.0,
        'min_child_weight': 2,
        'max_depth': 3,
#        'num_round': 10
    }


@model_decorator
def subsample():
    return {
        'subsample': 0.5,
        'gamma': 1.0,
        'max_depth': 5,
    }


@model_decorator
def colsubsample():
    return {
        'colsubsample_bytree': 0.5,
        'gamma': 1.0,
        'max_depth': 5,
    }


@model_decorator
def subcolsub():
    return {
        'subsample': 0.5,
        'colsubsample_bysplit': 0.3,
        'gamma': 1.0,
        'max_depth': 5,
    }


@model_decorator
def imbalance():
    return {
        'max_delta_step': 1.0,
        'colsubsample_bytree': 0.75,
        'gamma': 1.0,
        'max_depth': 5,
    }


@feat_decorator
def feats0(): return 11, ['loc', 'stats', 'hog']


@feat_decorator
def feats1(): return 11, ['loc', 'col', 'stats', 'hog']


@feat_decorator
def feats2(): return 7, ['loc', 'col', 'stats', 'kpolar', 'kmeancol']


@feat_decorator
def feats3(): return 11, ['loc', 'col', 'stats', 'kpolar']


@feat_decorator
def feats4(): return 11, ['loc', 'col', 'stats', 'kpolar', 'kmeandiff']


@feat_decorator
def feats5(): return 11, ['kmeandiff']


@feat_decorator
def feats6(): return 11, ['loc', 'kpolar']


@feat_decorator
def feats7(): return 11, ['loc', 'kpolar', 'kmeandiff']


@feat_decorator
def feats8(): return 11, ['loc', 'col', 'stats', 'histdiff']


@feat_decorator
def feats00(): return 11, ['loc', 'col']
