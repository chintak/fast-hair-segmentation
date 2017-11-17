import argparse
import configs
import multiprocessing as mp
import numpy as np
import os
from skimage.io import imread, imsave
import skimage.color as color
from skimage.segmentation import find_boundaries
import xgboost as xgb

import data
from configs import HAIR, FACE, BKG
from utils import hr_name_to_models, hr_predict_single


def name_to_viz_name(name, gt):
    return os.path.join('viz{}'.format('_gt/' if gt else '/'), os.path.basename(name))


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


def viz_png_file(im, mask):
    img = im.copy()
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maskf = (find_boundaries(mask==FACE, mode='inner') > 0).astype(np.uint8)
    r[maskf==1], g[maskf==1], b[maskf==1] = 0, 255, 0
    maskh = (find_boundaries(mask==HAIR, mode='inner') > 0).astype(np.uint8)
    r[maskh==1], g[maskh==1], b[maskh==1] = 255, 0, 255
    img = np.dstack((r, g, b))
    return img


def viz_files(names, keyps, model_fname, png=True):
    model_confs = hr_name_to_models(model_fname)
    for k, (name, keyp) in enumerate(zip(names, keyps)):
        if not os.path.exists(name): continue
        im = imread(name)
        prs = hr_predict_single(im, keyp, model_confs, overlap=0.5, last=False)
        if png:
            pr_imgs = [viz_png_file(im, pr) for pr in prs]
            imsave(name_to_viz_name(name, False), pr_imgs[-1])
            gt_mask = data.img2gt(name)
            gt_img = viz_png_file(im, gt_mask)
            imsave(name_to_viz_name(name, True), gt_img)
#        if k % 10 == 0: print "[{}] Done {}".format(os.getpid(), k)
    if not png:
        return im, prs


def visualize(model_fname, mat_viz_file):
    model_type, feat_type = get_model_feature_type(model_fname)
    bst = xgb.Booster(params=getattr(configs, model_type)(val=True))
    bst.load_model(model_fname)
    Conf = getattr(configs, feat_type)()
    featurize = Conf['FEATS']
    window = Conf['WINDOW']

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
                             bst.copy(), featurize, window))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

def args():
    args = argparse.ArgumentParser()
    args.add_argument('model_file', help='')
    args.add_argument('mat_file', help='')
    return args.parse_args()


if __name__ == '__main__':
    parse = args()
    visualize(parse.model_file, parse.mat_file)
