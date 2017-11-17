import argparse
import configs
import os
from glob import glob
import data
from visualize import viz_files, viz_png_file
from skimage.io import imsave
import xgboost as xgb

from utils import get_model_feature_type
from utils import hr_name_to_models, hr_predict_single


files_to_show = [
    'Adolfo_Aguilar_Zinser_0001',
    'Ai_Sugiyama_0002',
    'Ahmed_Ahmed_0001',
    'Alexander_Payne_0001',
]

save_prefix = 'benchmarks/'


def main(model_fnames):
    names, keypoints = data.mat_to_name_keyp('LFW/FaceKeypointsHOG_11_Test.mat')
    nshow, kpshow = [], []
    for nm, keyp in zip(names, keypoints):
        n, _ = os.path.splitext(os.path.basename(nm))
        if n in files_to_show:
            nshow.append(nm)
            kpshow.append(keyp)

    for mname in model_fnames:
        for i, (nm, keyp) in enumerate(zip(nshow, kpshow)):
            im, masks = viz_files([nm], [keyp], mname, png=False)
            for m, mnm in zip(masks, mname.split(',')):
                mim = viz_png_file(im, m)
                mnm, _ = os.path.splitext(os.path.basename(mnm))
                onm = os.path.join(save_prefix, '{}_{}.jpg'.format(mnm, i))
                imsave(onm, mim)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('mname', nargs='+', help='Model file names')
    parser = args.parse_args()

    if parser.mname:
        model_fnames = [','.join(parser.mname)]
    else:
        model_fnames = glob('models/*.model')
    print "Model: {}".format(model_fnames)
    main(model_fnames)
