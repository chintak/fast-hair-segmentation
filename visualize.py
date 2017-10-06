import argparse
import configs
import multiprocessing as mp
import numpy as np
import os
from skimage.io import imread, imsave
import skimage.color as color
import xgboost as xgb

import data
from configs import HAIR, FACE, BKG
from configs import Conf

WINDOW = Conf['WINDOW']

def name_to_viz_name(name, gt):
    return os.path.join('viz{}'.format('_gt/' if gt else '/'), os.path.basename(name))


def viz_png_file(im, mask, name, gt=False):
    img = im.copy()
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    if not gt:
        r[mask==HAIR] = 220
        b[mask==FACE] = 220
    else:
        r[mask==HAIR], g[mask==HAIR], b[mask==HAIR] = 0, 0, 255
        r[mask==FACE], g[mask==FACE], b[mask==FACE] = 0, 255, 0
    # r[mask==BKG],  g[mask==BKG],  b[mask==BKG]  = 255, 0, 0
    img = np.dstack((r, g, b))
    imsave(name_to_viz_name(name, gt), img)


def viz_files(names, keyps, bst, png=True):
    featurize = Conf['FEATS']
    for k, (name, keyp) in enumerate(zip(names, keyps)):
        if not os.path.exists(name): continue
        im = imread(name)
        idxs, patches = data.patchify(im, WINDOW)
        feats = np.asarray([featurize.process(x, y, patch, im, keyp)
                            for (x, y), patch in zip(idxs, patches)], dtype=np.float)
        dset = xgb.DMatrix(feats)
        preds = np.argmax(bst.predict(dset), axis=1)
        pr = data.unpatchify(im.shape, idxs, preds, WINDOW)
        if png: viz_png_file(im, pr, name)
        if png:
            gt_mask = data.img2gt(name)
            viz_png_file(im, gt_mask, name, gt=True)
        if k % 10 == 0: print "[{}] Done {}".format(os.getpid(), k)
    if not png:
        return im, pr

def visualize(model_fname, mat_viz_file):
    bst = xgb.Booster(params=configs.basic(val=True))
    bst.load_model(model_fname)

    names, keypoints = data.mat_to_name_keyp(mat_viz_file)

    NUM_TRAIN_SAMPS = 100  # len(names)
    nprocs = mp.cpu_count()
    chunksize = int(NUM_TRAIN_SAMPS // nprocs)
    procs = []

    for i in range(nprocs):
        lim = chunksize * (i+1) if i < nprocs - 1 else NUM_TRAIN_SAMPS
        p = mp.Process(target=viz_files,
                       args=(names[chunksize*i:lim],
                             keypoints[chunksize*i:lim],
                             bst.copy()))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

def args():
    args = argparse.ArgumentParser()
    args.add_argument('model_file', help='')
    args.add_argument('mat_file', help='')
    args.add_argument('-w', '--window', help='window size')
    return args.parse_args()


if __name__ == '__main__':
    parse = args()
    WINDOW = parse.window if parse.window else WINDOW
    visualize(parse.model_file, parse.mat_file)
