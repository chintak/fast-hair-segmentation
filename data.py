import argparse
import cPickle as pickle
from glob import glob
import multiprocessing as mp
import numpy as np
import os
from scipy.io import loadmat
from skimage.io import imread
from sklearn.datasets import dump_svmlight_file
from subprocess import call
import sys
import xgboost as xgb
import tempfile as tm

import configs
from configs import HAIR, FACE, BKG

Conf = None
Set = ''
WINDOW = None
DATA_DIR = 'LFW/'
tmdir = tm.mkdtemp(dir=DATA_DIR)


def img2gt(name):
    datf = os.path.join('LFW/parts_lfw_funneled_gt',
                        '/'.join(name.split('.')[0].split('/')[-2:]) + '.dat')
    supf = os.path.join('LFW/parts_lfw_funneled_superpixels_mat',
                        '/'.join(name.split('.')[0].split('/')[-2:]) + '.dat')
    gt = np.loadtxt(datf, dtype=np.uint8)[1:]
    gtsup = np.loadtxt(supf, dtype=np.uint64)
    return np.apply_along_axis(lambda row: map(lambda k: gt[k], row), 1, gtsup)


def gen_train_data(proc_names, proc_keypoints, out_q=None, seed=1234):
    if not Conf or not WINDOW: raise ValueError('Conf not set')
    featurize = Conf['FEATS']
    num_feats = 0
    for i, nm in enumerate(proc_names):
        if os.path.exists(nm):
            t1 = imread(nm)
            tk1 = proc_keypoints[i]
            num_feats = len(featurize.process(
                0, 0, t1[0:WINDOW, 0:WINDOW, :], t1, tk1))
            break
    if not num_feats: return

    M, N, _ = t1.shape
    xsr, ysr = sampling(M, N, WINDOW)
    batch_samples = min(100, len(proc_names)) * len(xsr)
    train_x = np.zeros((batch_samples, num_feats), dtype=np.float)
    train_y = np.zeros((batch_samples,), dtype=np.uint8)
    name = os.path.join(tmdir,
                        'hair.{}.txt.{}'.format(os.getpid(), Set.lower()))
    print '[{}] # patches/img = {} Feature size = {}'.format(
        os.getpid(), len(xsr), num_feats)
    print '[{}] Writing to {}'.format(os.getpid(), name)
    fp = open(name, 'wb')
    j = 0
    np.random.seed(seed)

    for i, (fn, keyp) in enumerate(zip(proc_names, proc_keypoints)):
        if not os.path.exists(fn):
            print "Errr {} {}".format(i, fn)
            continue
        im = imread(fn)
        gt = img2gt(fn)
        m, n, _ = im.shape
        # sample patches
        if m != M or n != N:
            xsr, ysr = sampling(m, n, WINDOW)
            M, N = m, n
        if np.any(keyp[:-1, :] > np.asarray([m - WINDOW//2, n - WINDOW//2])):
            print '[{}] Skipping {}'.format(os.getpid(), fn)
            continue
        for x, y in zip(xsr, ysr):
            patch = im[y:y+WINDOW, x:x+WINDOW, :]
            gtpatch = gt[y:y+WINDOW, x:x+WINDOW]
            train_x[j, :] = featurize.process(x, y, patch, im, keyp)
            train_y[j] = featurize.processY(gtpatch)
            j += 1
        if j == train_x.shape[0]:
            dump_svmlight_file(train_x, train_y, fp)
            j = 0
        if i % 10 == 0:
            print "[{}] Done {}.".format(os.getpid(), i)

    if j: dump_svmlight_file(train_x[:j, :], train_y[:j], fp)
    fp.close()
    if out_q:
        out_q.put(name)
    else:
        return train_x, train_y


def sampling(m, n, window):
    xs, ys = np.meshgrid(range(0, n - window, window),
                         range(0, m - window, window))
    xsr, ysr = xs.ravel(), ys.ravel()
    return (xsr, ysr)


def count_samples(ys):
    n_face, n_hair, n_bkg = 0, 0, 0
    for e in ys:
        n_hair += (e==HAIR)
        n_face += (e==FACE)
        n_bkg += (e==BKG)
    return n_hair, n_face, n_bkg


def patchify(im, window):
    m, n, _ = im.shape
    xsr, ysr = sampling(m, n, window)
    patches = [im[y:y+window, x:x+window, :] for x, y in zip(xsr, ysr)]
    return zip(xsr, ysr), patches


def unpatchify(shape, idxs, preds, window):
    m, n, _ = shape
    im = np.zeros((m, n)) + BKG
    for (x, y), pr in zip(idxs, preds):
        im[y:y+window, x:x+window] = pr
    return im


def mat_to_name_keyp(mat_file):
    kp = loadmat(mat_file)
    keypoints = map(lambda k: k[0] - 1, kp['Keypoints'])
    names = map(lambda k: 'LFW/lfw_funneled/' + '/'.join(str(k[0][0]).split('/')[-2:]), kp['Names'])
    # hogs = map(lambda k: k[0], kp['HOGs'])
    return names, keypoints


def main():
    global Set, Conf, WINDOW
    args = argparse.ArgumentParser()
    args.add_argument('set', help='Train, Test or Validation')
    args.add_argument('suf', help='hair_<window>_<suf>.txt.<set>')
    parse = args.parse_args()
    Conf = getattr(configs, parse.suf)()
    Set = parse.set
    WINDOW = Conf['WINDOW']

    print "Using {} set".format(Set)
    train_mat_path = 'LFW/FaceKeypointsHOG_11_{}.mat'.format(Set)
    names, keypoints = mat_to_name_keyp(train_mat_path)

    NUM_TRAIN_SAMPS = len(names)

    nprocs = mp.cpu_count()
    out_q = mp.Queue()
    chunksize = int(NUM_TRAIN_SAMPS // nprocs)
    procs = []
    rngs = np.random.randint(0, 100000, size=(nprocs,))

    print "Total training samples: {}. Starting {} procs with chunksize of {}.".format(
        NUM_TRAIN_SAMPS, nprocs, chunksize)

    for i in range(nprocs):
        lim = chunksize * (i+1) if i < nprocs - 1 else NUM_TRAIN_SAMPS
        p = mp.Process(target=gen_train_data,
                       args=(names[chunksize*i:lim],
                             keypoints[chunksize*i:lim],
                             out_q, rngs[i]))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print "Joined processes"

    trainN = []
    for i in range(nprocs):
        n = out_q.get()
        trainN.append(n)

    # merge proc train files into 1
    cmd = ["cat"]
    cmd.extend(trainN)
    cmd.append('>')
    cmd.append(os.path.join(
        DATA_DIR, 'hair_{}.txt.{}'.format(parse.suf, Set.lower())))
    call(' '.join(cmd), shell=True)

    print "Done"

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print e
    # clean up
    cmd = ["rm", "-rf", tmdir]
    call(' '.join(cmd), shell=True)
