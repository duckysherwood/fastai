from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings


def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max(
            [
                fbeta_score(targs, (preds > th), 2, average="samples")
                for th in np.arange(start, end, step)
            ]
        )


def opt_th(preds, targs, start=0.17, end=0.24, step=0.01):
    ths = np.arange(start, end, step)
    idx = np.argmax(
        [fbeta_score(targs, (preds > th), 2, average="samples") for th in ths]
    )
    return ths[idx]


def get_data(path, transformations, batch_size, n, cv_idx):
    val_idxs = get_idxs_for_validation_split(n, cv_idx)
    return ImageClassifierData.from_csv(
        path,
        "train-jpg",
        f"{path}train_v2.csv",
        batch_size,
        transformations,
        suffix=".jpg",
        val_idxs=val_idxs,
        test_name="test-jpg",
    )


def get_data_zoom(f_model, path, image_size, batch_size, n, cv_idx):
    transformations = transformations_from_model(
        f_model, image_size, aug_transformations=transforms_top_down, max_zoom=1.05
    )
    return get_data(path, transformations, batch_size, n, cv_idx)


def get_data_pad(f_model, path, image_size, batch_size, n, cv_idx):
    transforms_pt = [
        RandomRotateZoom(9, 0.18, 0.1),
        RandomLighting(0.05, 0.1),
        RandomDihedral(),
    ]
    transformations = transformations_from_model(f_model, image_size, aug_transformations=transforms_pt, pad=image_size // 12)
    return get_data(path, transformations, batch_size, n, cv_idx)
