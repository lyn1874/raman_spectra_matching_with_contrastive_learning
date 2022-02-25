"""
Created on 09:42 at 01/06/2021
@author: bo 
"""
import numpy as np
from scipy.interpolate import interp1d


def find_interp(spectrum, target_wave):
    """Interpolate the spectrum with respect to the fixed wave interval
    Args:
        spectrum: [N, 2]
        target_wave: [N]
    """
    if len(spectrum) == 2:
        func = interp1d(spectrum[0], spectrum[1], kind="slinear", bounds_error=False, fill_value=0)
    elif np.shape(spectrum)[1] == 2:
        func = interp1d(spectrum[:, 0], spectrum[:, 1], kind="slinear", bounds_error=False, fill_value=0)
    target_intensity = func(target_wave)
    return target_intensity


def norm(value, method="max_1_min_0"):
    if method == "max_1":
        return value / value.max()
    elif method == "max_1_min_0":
        return (value - np.min(value)) / (np.max(value) - np.min(value))
    elif method == "abs_max_1_min_0":
        value = abs(value)
        return (value - np.min(value)) / (np.max(value) - np.min(value))
    elif method == "mean_0_std_1":
        return (value - np.mean(value)) / np.std(value)
    elif method == "abs":
        return abs(value)
    else:
        return value


def get_target_data_with_randomleaveone(path, preprocess, random_leave_one_out):
    """This function gives the training spectra and testing spectra depends on whether we are going to do random
    leave one out"""
    if preprocess:
        path += "_preprocess/"
    else:
        path += "_raw/"
    if preprocess:
        xy_train = np.load(path + "scale_xy_train.npy")
        gt_train = np.load(path + "yclass_train.npy")
        xy_val = np.load(path + "scale_xy_val.npy")
        gt_val = np.load(path + "yclass_val.npy")
        if random_leave_one_out:
            spectra_tot = np.concatenate([xy_train, xy_val], axis=0)
            label_tot = np.concatenate([gt_train, gt_val], axis=0)
            unique_cls = np.unique(label_tot)
            tr_index = []
            for i in unique_cls:
                index = np.where(label_tot == i)[0]
                _tr = np.random.choice(index, len(index) - 1, replace=False)
                tr_index.append(_tr)
            tr_index_array = np.array([v for j in tr_index for v in j])
            tt_index_array = np.delete(np.arange(len(spectra_tot)), tr_index_array)
            xy_train_rand = spectra_tot[tr_index_array]
            xy_test_rand = spectra_tot[tt_index_array]
            return [xy_train_rand, label_tot[tr_index_array]], [xy_test_rand, label_tot[tt_index_array]]
        else:
            return [xy_train, gt_train], [xy_val, gt_val]
    else:
        interpolate = "interpolate_norm"
        xy = np.load(path + "xy.npy", allow_pickle=True)
        xx = np.load(path + "xx.npy", allow_pickle=True)
        if "interpolate" in interpolate:
            target_wave = np.linspace(200, 3700, 1100)
            xy_interpolate = [find_interp([xx[i], xy[i]], target_wave)[:1024] for i in range(len(xx))]
            if "norm" in interpolate:
                xy_interpolate = [norm(x, "max_1_min_0") for x in xy_interpolate]
            xy = np.array(xy_interpolate)
        yclass = np.load(path + "yclass.npy")
        unique_class = np.unique(yclass)
        tr_index = []
        for i in unique_class:
            _index = np.where(yclass == i)[0]
            if random_leave_one_out:
                _select = np.random.choice(_index, len(_index) - 1, replace=False)
            else:
                _select = _index[:-1]
            tr_index.append(_select)
        tr_index = np.array([v for j in tr_index for v in j])
        tt_index = np.delete(np.arange(len(xy)), tr_index)
        return [xy[tr_index], yclass[tr_index]], [xy[tt_index], yclass[tt_index]]












