import os
import numpy as np
import xgboost as xgb

import configs
from configs import BKG


def sampling(m, n, window, overlap=0):
    freq = int(round(window * (1. - overlap)))
    xs, ys = np.meshgrid(range(0, n - window, freq),
                         range(0, m - window, freq))
    xsr, ysr = xs.ravel(), ys.ravel()
    return (xsr, ysr)


def patchify(im, window, overlap=0):
    m, n, _ = im.shape
    xsr, ysr = sampling(m, n, window, overlap)
    patches = [im[y:y+window, x:x+window, :] for x, y in zip(xsr, ysr)]
    return zip(xsr, ysr), patches


def unpatchify(shape, idxs, preds, window):
    m, n, _ = shape
    im = np.zeros((m, n, 3))
    mk = np.zeros((m, n))
    for (x, y), pr in zip(idxs, preds):
        im[y:y+window, x:x+window, pr] += 1
        mk[y:y+window, x:x+window] += 1.
    im = np.argmax(im, axis=2)
    im[mk == 0] = BKG
    return im


def hr_name_to_models(mnames):
    model_feats = []
    if not mnames: return []
    for model_fname in mnames.split(','):
        model_type, feat_type = get_model_feature_type(model_fname)
        bst = xgb.Booster(params=getattr(configs, model_type)(val=True))
        bst.load_model(model_fname)
        Conf = getattr(configs, feat_type)()
        featurize = Conf['FEATS']
        window = Conf['WINDOW']
        model_feats.append((featurize, bst, window))
    return model_feats


def _predict(feats, bst):
    dset = xgb.DMatrix(feats)
    return np.argmax(bst.predict(dset), axis=1)


def predict_single(im, keyp, featurize, bst, window, overlap=0, hr_maps=[]):
    idxs, patches = patchify(im, window, overlap)
    feats = np.asarray([featurize.process(x, y, patch, im, keyp, hr_maps)
                        for (x, y), patch in zip(idxs, patches)], dtype=np.float)
    preds = _predict(feats, bst)
    return unpatchify(im.shape, idxs, preds, window)


def hr_predict_single(im, keyp, hr_model_feats, overlap=0, last=True):
    if not hr_model_feats: return None
    hr_map = []
    res = []
    for featurize, bst, window in hr_model_feats:
        res.append(predict_single(im, keyp, featurize, bst, window,
                                  overlap, hr_map))
        hr_map = [res[-1]]
    return res[-1] if last else res


def img2gt(name):
    datf = os.path.join('LFW/parts_lfw_funneled_gt',
                        '/'.join(name.split('.')[0].split('/')[-2:]) + '.dat')
    supf = os.path.join('LFW/parts_lfw_funneled_superpixels_mat',
                        '/'.join(name.split('.')[0].split('/')[-2:]) + '.dat')
    gt = np.loadtxt(datf, dtype=np.uint8)[1:]
    gtsup = np.loadtxt(supf, dtype=np.uint64)
    return np.apply_along_axis(lambda row: map(lambda k: gt[k], row), 1, gtsup)


def get_model_feature_type(name):
    n, e = os.path.splitext(os.path.basename(name))
    mtype, ftype = n.split('_')[:2]
    if not hasattr(configs, mtype):
        print 'Invalid model type in {}. See configs.py'.format(name)
        mtype, ftype = None, None
    if not hasattr(configs, ftype):
        print 'Invalid feature type in {}. See configs.py'.format(name)
        mtype, ftype = None, None
    return mtype, ftype
