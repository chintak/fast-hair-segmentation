import numpy as np
import xgboost as xgb

import data


def _predict(feats, bst):
    dset = xgb.DMatrix(feats)
    return np.argmax(bst.predict(dset), axis=1)


def predict_single(im, keyp, featurize, bst, window, overlap=1.0):
    idxs, patches = data.patchify(im, window, overlap)
    feats = np.asarray([featurize.process(x, y, patch, im, keyp)
                        for (x, y), patch in zip(idxs, patches)], dtype=np.float)
    preds = _predict(feats, bst)
    return data.unpatchify(im.shape, idxs, preds, window)
