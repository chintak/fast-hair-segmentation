import configs
import os
from glob import glob
import data
from visualize import viz_files
from skimage.io import imsave


files_to_show = [
    'Aaron_Sorkin_0002.jpg',
    'Adolfo_Aguilar_Zinser_0001',
    'Abdoulaye_Wade_0004.jpg',
    'Alexandra_Vodjanikova_0002',
    'Alice_Fisher_0002',
]

save_prefix = 'benchmarks/'

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


def main():
    model_fnames = glob('models/*.model')
    names, keypoints = data.mat_to_name_keyp('LFW/FaceKeypointsHOG_11_Test.mat')
    nshow, kpshow = [], []
    for nm, keyp in zip(names, keypoints):
        n = os.path.basename(nm)
        if n in files_to_show:
            nshow.append(nm)
            kpshow.append(keyp)

    for mname in model_fnames:
        model_type, feat_type = get_model_feature_type(mname)
        if not model_type: continue
        bst = xgb.Booster(params=getattr(configs, model_type)(val=True))
        bst.load_model(mname)
        Conf = getattr(configs, feat_type)()
        featurize = Conf['FEATS']
        window = Conf['WINDOW']
        for i, nm, keyp in enumerate(zip(nshow, kpshow)):
            _, mim = viz_files([nm], [keyp], bst, featurize, window, png=False)
            mnm, _ = os.path.splitext(os.path.basename(mname))
            onm = os.path.join(save_prefix, '{}_{}.jpg'.format(mnm, i))
            imsave(onm, mim)


if __name__ == '__main__':
    main()
