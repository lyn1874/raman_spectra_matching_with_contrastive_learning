"""
Created on 12:10 at 09/07/2021
@author: bo 
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import pickle
import seaborn as sns
from scipy.interpolate import interp1d


def get_spectrum(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = data[10:-5]
    data_array = [np.array(v.split("\n")[0].split(",")).astype("float32") for v in data]
    return np.array(data_array)


def get_minerals_siamese(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = np.array([v.split("\n")[0] for v in data])
    return data


def convert_chemical(v):
    c = v.replace("+^", "+}").replace("^", "^{")
    cc = c.split("_")
    for i, _cc in enumerate(cc):
        try:
            _dd = int(_cc[0])
            _cc = "_{" + _cc + "}"
        except ValueError:
            if i != 0 and i != len(cc) - 1:
                if "^" in _cc:
                    _cc = _cc + "}"
        cc[i] = _cc
    value = "".join(cc)
    value = value.replace("&#183;", '')
    if "&#" in value:
        value = value.split(" ")[0]
    value = value.replace("-^{}", "}").replace("-^{", "}")
    value = value.replace("}}", "}").replace("{{", "{")  # .replace("-^{}", "}").replace("-^{", "}")
    return value


def get_chemical(path):
    with open(path, 'r') as f:
        data = f.readlines()
    ch = data[2].split("=")[1].split("\n")[0]
    if ch[-1] == "_":
        ch = ch[:-1]
    ch = convert_chemical(ch)
    return ch


def give_all_raw(path2read="../rs_dataset/", print_info=True):
    siamese_minerals = get_minerals_siamese(path2read + "minerals.txt")
    path_0 = path2read + "unrated_unoriented_unoriented.csv"
    path_1 = path2read + "poor_unoriented_unoriented.csv"
    path_2 = path2read + "fair_unoriented_unoriented.csv"
    path_3 = path2read + "ignore_unoriented_unoriented.csv"
    path_4 = path2read + "excellent_unoriented_unoriented_raw.csv"
    for i, path in enumerate([path_0, path_1, path_2, path_3, path_4]):
        _data = give_subset_of_spectrums(path, None, "raw", print_info=print_info)
        if i == 0:
            data = _data
        else:
            data = pd.concat([data, _data])
    name = np.array([v.lower() for v in data.name.to_numpy()])
    select_index = [iterr for iterr, v in enumerate(name) if v in siamese_minerals]
    data_subset = data.iloc[select_index]
    names = data_subset.name.to_numpy()
    unique_name, unique_count = np.unique(names, return_counts=True)
    if print_info:
        print("There are %d unique minerals with %d spectra" % (len(unique_name), len(names)))
        print("with the smallest number of spectra: %d and largest number of spectra %d" % (np.min(unique_count),
                                                                                            np.max(unique_count)))
    return data_subset


def give_subset_of_spectrums(path, laser=None, raw_preprocess="raw", unrate=False, print_info=True):
    """Give a subset of spectrums
    Args:
        path: the csv file for all the data
        laser: int, for the low resolution:either 532, 785 or 780, for the excellent resolution: 532, 780, 514, 785
        raw_preprocess: int, either 0 or 1, where 1 corresponds to raw data and 0 corresponds to preprocessed data
    """
    data = pd.read_csv(path)
    num_original = len(data)
    unique_original = len(data["name"].unique())
    if laser:
        data = data[data["laser"] == laser]
    if unrate:
        data_5 = data[data["laser"] == 532]
        data_7 = data[data["laser"] == 785]
        data_72 = data[data["laser"] == 780]
        data = pd.concat([data_5, data_7, data_72], axis=0)
    if raw_preprocess != "total":
        raw_preprocess = [0 if raw_preprocess == "preprocess" else 1][0]
        if print_info:
            print("Filter out raw spectrums")
        data = data[data["raw"] == raw_preprocess]
    unique_name = len(data["name"].unique())
    if print_info:
        print("........Select %d data points out of %d data points............." % (len(data), num_original))
        print("........Select %d classes out of %d classes....................." % (unique_name, unique_original))
    return data


def create_label(name):
    unique_name = np.unique(name)
    label = np.arange(len(name))
    for i, single_name in enumerate(unique_name):
        _index = np.where(name == single_name)[0]
        label[_index] = i
    return label, unique_name


def show_mineral(tr_filename, tt_filename, path_mom, show_specific="Actinolite", save=False):
    tr_shortname = np.array([v.split("__")[0] for v in tr_filename])
    tt_shortname = np.array([v.split("__")[0] for v in tt_filename])
    if "rs_dataset" in tr_shortname[0]:
        tr_shortname = np.array([v.split("unoriented/")[1] for v in tr_shortname])
        tt_shortname = np.array([v.split("unoriented/")[1] for v in tt_shortname])
    rand_ = np.array([v for v in tt_shortname if v in tr_shortname])
    select = np.random.choice(rand_, [1])[0]
    _tt_select = tt_filename[np.where(tt_shortname == select)[0]]
    _tr_select = tr_filename[np.where(tr_shortname == select)[0]]
    _non = np.array([v for v in tt_shortname if v not in tr_shortname])
    if len(_non) == 0:
        tot_plot = 2
    else:
        tot_plot = 3
    if show_specific != "None":
        _tr_ac_select = tr_filename[np.where(tr_shortname == show_specific)[0]]
        _tt_ac_select = tt_filename[np.where(tt_shortname == show_specific)[0]]
        _tr_ac_select = np.concatenate([_tr_ac_select, _tt_ac_select], axis=0)
        with sns.axes_style("darkgrid"):
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            nm_value = [v.split("__")[1] + " nm:" + v.split("__")[3] for v in _tr_ac_select]
            for _s in _tr_ac_select:
                if "rs_dataset" not in _s:
                    _s = path_mom + _s
                spec = get_spectrum(_s)
                ax.plot(spec[:, 0], norm(spec[:, 1], "max_1_min_0"), lw=1)
            ax.legend(nm_value, fontsize=7, loc='best')
            ax.set_xlabel("Raman shift", fontsize=7)
            ax.set_ylabel("Intensity", fontsize=7)
            if save:
                plt.savefig("../rs_dataset/%s_spectrum.pdf" % show_specific, pad_inches=0, bbox_inches='tight')
    fig = plt.figure(figsize=(10, 5))

    for i, ss in enumerate([_tr_select, _tt_select]):
        ax = fig.add_subplot(1, tot_plot, i + 1)
        for _ss in ss:
            if "rs_dataset" not in _ss:
                _ss = path_mom + _ss
            spec = get_spectrum(_ss)
            ax.plot(spec[:, 0], norm(spec[:, 1], "max_1_min_0"), lw=1)
        ax.legend(ss, fontsize=6, loc='best')
    if tot_plot == 3:
        ax = fig.add_subplot(133)
        for i, s in enumerate(np.random.choice(_non, 5, replace=False)):
            s = np.where(tt_shortname == s)[0][0]
            s = tt_filename[s]
            if "rs_dataset" not in s:
                s = path_mom + s
            spec = get_spectrum(s)
            ax.plot(spec[:, 0], norm(spec[:, 1], "max_1_min_0"))
        ax.legend(_non, fontsize=6, loc='best')


def set_label_from_zero(tr_label, tt_label, label_tot):
    label_correspond_tr = label_tot[tr_label]
    label_correspond_tt = label_tot[tt_label]
    label_unique = np.unique(label_correspond_tr)
    tr_update = np.zeros_like(tr_label)
    tt_update = np.zeros_like(tt_label)
    for j, s in enumerate(label_unique):
        l = np.where(label_correspond_tr == s)[0]
        tr_update[l] = j
        lt = np.where(label_correspond_tt == s)[0]
        tt_update[lt] = j
    update_tr_label = label_unique[tr_update]
    original_tr_label = label_tot[tr_label]
    diff = [v == q for v, q in zip(update_tr_label, label_correspond_tr)]
    # print("Value %d should equal to value %d, otherwise WRONG!!!" % (np.sum(diff), len(update_tr_label)))
    update_tt_label = label_unique[tt_update]
    diff = [v == q for v, q in zip(update_tt_label, label_correspond_tt)]
    # print("Value %d should equal to value %d, otherwise WRONG!!!" % (np.sum(diff), len(update_tt_label)))
    return label_unique, tr_update, tt_update


def mini_dataset(data, freq, num_tt, show=False, path_mom=None, test=False, show_specific="None", save=False,
                 random_leave_one_out=False, print_info=True):
    """Creates mini-mineral dataset. I only consider the ones that have number of spectrums > than the specified
    freq"""
    filename = data.full_name.to_numpy()
    name = data.name.to_numpy()
    label, label_name = create_label(name)
    unique_name, unique_count = np.unique(name, return_counts=True)
    select_index = np.where(unique_count >= freq)[0]
    if print_info:
        print("There are %d classes in the dataset with %d spectrums" % (len(select_index),
                                                                         np.sum(unique_count[select_index])))
    tr_index, tt_index = [], []
    for i in select_index:
        _name = unique_name[i]
        _act_index = np.where(name == _name)[0]
        if not random_leave_one_out:
            tr_index.append(_act_index[:-num_tt])  # fix the test dataset and then check the experiment
            tt_index.append(_act_index[-num_tt:])
        else:
            _tr_select = np.random.choice(np.arange(len(_act_index)), len(_act_index) - 1, replace=False)
            tr_index.append(_act_index[_tr_select])
            tt_index.append(_act_index[np.delete(np.arange(len(_act_index)), _tr_select)])
    tr_index = np.array([v for j in tr_index for v in j])
    tt_index = np.array([v for j in tt_index for v in j])
    tr_filename, tr_label = filename[tr_index], label[tr_index]
    tt_filename, tt_label = filename[tt_index], label[tt_index]
    label_subset, tr_label_update, tt_label_update = set_label_from_zero(tr_label, tt_label, label_name)
    if show:
        show_mineral(tr_filename, tt_filename, path_mom, show_specific, save)
    if not test:
        return tr_filename, tt_filename, tr_label_update, tt_label_update, label_subset, tr_index, tt_index
    else:
        rest = np.delete(np.arange(len(name)), np.concatenate([tr_index, tt_index], axis=0))
        rest_filename = filename[rest]
        rest_act_label = label[rest]
        rest_label = np.zeros([len(rest_filename)])
        rest_unique_label, rest_unique_count = np.unique(rest_act_label, return_counts=True)
        label_exist = np.max(tt_label_update) + 1
        for j, u_l in enumerate(rest_unique_label):
            _index = np.where(rest_act_label == u_l)[0]
            rest_label[_index] = label_exist
            label_exist += 1
        label_name_tot = np.concatenate([label_subset, label_name[label[rest]]], axis=0)
        return tr_filename, tt_filename, rest_filename, tr_label_update, tt_label_update, rest_label, label_subset, label_name_tot


def find_interp(spectrum, target_wave):
    """Interpolate the spectrum with respect to the fixed wave interval
    Args:
        spectrum: [N, 2]
        target_wave: [N]
    """
    if len(np.shape(spectrum)) != 1:
        if len(spectrum) == 2:
            func = interp1d(spectrum[0], spectrum[1], kind="slinear", bounds_error=False, fill_value=0)
        elif np.shape(spectrum)[1] == 2:
            func = interp1d(spectrum[:, 0], spectrum[:, 1], kind="slinear", bounds_error=False, fill_value=0)
        target_intensity = func(target_wave)
    else:
        target_intensity = spectrum
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


def find_pos_neg_pair_fast(label):
    unique_label = np.unique(label)
    neg_pair, pos_pair = [], []
    for i in unique_label:
        _ind = np.where(label == i)[0]
        ind_not = np.where(label[np.max(_ind):] != i)[0] + np.max(_ind)
        _pos_ind = np.triu_indices(len(_ind), 1)
        _pos_ind = np.concatenate([np.expand_dims(_pos_ind[0], axis=1),
                                   np.expand_dims(_pos_ind[1], axis=1)], axis=1)
        pos_pair.append(_pos_ind + np.min(_ind))
        if i != np.max(unique_label):
            _neg_ind = [np.concatenate([[np.repeat(j, len(ind_not))],
                                        [ind_not]], axis=0) for j in _ind]
            _neg_ind = np.reshape(np.transpose(_neg_ind, (0, 2, 1)), [-1, 2])
            neg_pair.append(_neg_ind)
    pos_pair = np.concatenate(pos_pair, axis=0)
    neg_pair = np.concatenate(neg_pair, axis=0)
    pos_pair = np.concatenate([pos_pair, label[pos_pair[:, :1]], label[pos_pair[:, 1:]], np.ones([len(pos_pair), 1])],
                              axis=-1).astype(np.int32)
    neg_pair = np.concatenate([neg_pair, label[neg_pair[:, :1]], label[neg_pair[:, 1:]], np.ones([len(neg_pair), 1])],
                              axis=-1).astype(np.int32)
    return pos_pair, neg_pair


def find_pos_neg_pair(label, real_or_fake, beta):
    """Find positive and negative pairs of data
    Args:
        label: [num_samples], this label has to start from 0, and it needs to be consecutive!
        real_or_fake: [num_samples], 0 or 1, 0 means the spectrum is sampled from augmentation. 1 means the spectrum
        is the real spectrum
        beta: float number that determines how many negative pairs are selected
    """
    unique_label, label_count = np.unique(label, return_counts=True)
    pos_pair = []
    neg_pair = []
    for i, s_l in enumerate(label):
        step = i + 1
        if label_count[s_l] > 1:
            _pos = np.where(label[step:] == s_l)[0]
            pos_pair.append([[i, v + step, s_l, label[v + step], 1] for v in _pos])
            if real_or_fake[i] == 1:
                _neg_index = np.delete(np.arange(len(label) - step), _pos)
                use_or_not = np.random.binomial(1, beta, [len(_neg_index)])
                _neg_index = _neg_index[use_or_not == 1]
                neg_pair.append([[i, v + step, s_l, label[v + step], 0] for v in _neg_index])
        else:
            _neg_index = np.arange(len(label))[step:]
            use_or_not = np.random.binomial(1, beta, [len(_neg_index)])
            _neg_index = _neg_index[use_or_not == 1]
            neg_pair.append([[i, v, s_l, label[v], 0] for v in _neg_index])
    pos_pair = np.array([v for j in pos_pair for v in j])
    neg_pair = np.array([v for j in neg_pair for v in j])
    return pos_pair, neg_pair


def find_manu_pos_neg_pair(label, real_or_fake, beta=0.4):
    """Find positive and negative pairs of data
    Args:
        label: [num_samples], this label has to start from 0, and it needs to be consecutive!
        real_or_fake: [num_samples], 0 or 1, 0 means the spectrum is sampled from augmentation. 1 means the spectrum
        is the real spectrum
        beta: float, determines the probability of an actual negative pair being selected
    Ops:
    the difference between this func and the previous one is that
    here my negative pairs also consists of positive pairs,
    and I manually create hard negative by random shifting the spectra to left/right or manually changing the intensity
    """
    unique_label, label_count = np.unique(label, return_counts=True)
    pos_pair = []
    neg_pair = []
    for i, s_l in enumerate(label):
        step = i + 1
        if label_count[s_l] > 1:
            _pos = np.where(label[step:] == s_l)[0]
            pos_pair.append([[i, v + step, s_l, label[v + step], 1, 1] for v in _pos])
            _neg_index = np.delete(np.arange(len(label) - step), _pos)
            use_or_not = np.random.binomial(1, beta, [len(_neg_index)])
            _neg_index = _neg_index[use_or_not == 1]
            if real_or_fake[i] == 1 and len(_neg_index) > 0:
                neg_pair.append([[i, v + step, s_l, label[v + step], 1, 0] for v in _neg_index])
            neg_pair.append([[i, v + step, s_l, label[v + step], 0, 0] for v in _pos])
            # these are used for creating hard negative pairs
        else:
            _neg_index = np.arange(len(label))[step:]
            use_or_not = np.random.binomial(1, beta, [len(_neg_index)])
            _neg_index = _neg_index[use_or_not == 1]
            if len(_neg_index) > 0:
                neg_pair.append([[i, v, s_l, label[v], 1, 0] for v in _neg_index])
            # neg_pair.append([[i, v, s_l, label[v], 1, 0] for v in np.arange(len(label))[step:]])
            neg_pair.append([[i, i, s_l, s_l, 0, 0]])
    pos_pair = np.array([v for j in pos_pair for v in j])
    neg_pair = np.array([v for j in neg_pair for v in j])
    return pos_pair, neg_pair


def calc_std(x, k, mode="history"):
    """Calculate the standard deviation for the given spectrum x"""
    _std = []
    if mode != "pure_noise":
        if mode == "history":
            for i in range(len(x) - k):
                _std.append(np.std(np.diff(x[i:(i + k)])))
        elif mode == "future":
            for i in range(len(x))[k:]:
                _std.append(np.std(x[(i - k):i]))
        _std = np.concatenate([np.zeros([k // 2]), np.array(_std), np.zeros([k // 2])])
    else:
        _std = np.ones([len(x)])
    return _std


def random_move_npy(x, randshift, lower_bound=8, upper_bound=1480):
    num_orig = len(x)
    move = int(np.random.normal(1) * randshift)
    lower_ = lower_bound + move
    upper_ = upper_bound + move
    lower_ = [0 if lower_ < 0 else lower_][0]
    x = x[lower_:upper_]
    up = num_orig - lower_ - len(x)
    x = np.concatenate([np.zeros([lower_]), x, np.zeros([up])])
    return x


def aug_mineral_npy(num_sample, spec, k, randshift, lower_bound, upper_bound, mode="history", norm_std=1, check=False):
    """Augment the minerals that only have a single spectrum during training
    mode:
        pure_noise, means I only add noise on it
        history, means I add std * random noise on it
        repeat: I just simply repeat the spectra
    """
    if mode != "repeat":
        spec_std = calc_std(spec, k, mode)
        noise = np.random.normal(0, norm_std, [num_sample, len(spec)])
        spec = spec + noise * spec_std
    else:
        spec = np.repeat(np.expand_dims(spec, axis=0), num_sample, axis=0)
    if randshift != 0:
        spec_sample = []
        for i in range(num_sample):
            _spec = random_move_npy(spec[i], randshift, lower_bound=lower_bound,
                                    upper_bound=upper_bound)
            spec_sample.append(_spec)
        spec = spec_sample
    if check:
        return np.array(spec), spec_std
    return np.array(spec)


def sample_test(spectrum, norm_method):
    spectrum = [norm(v[1], norm_method) for v in spectrum]
    spectrum = np.array(spectrum)
    return spectrum


def sample(spectrum, label, tot_num, show=False, norm_first=False, norm_method="max_1_min_0", sample_new=True,
           lower_bound=10, upper_bound=1280, sliding_window=30, randshift=3, train_or_test="train",
           aug_more_than_one=False, noise_std=1, mode="history", print_info=True):
    """Sample more spectrum
    Args:
        spectrum: [M, 2, N], N wavenumber, M number spectrums from the same class
        label: [M, 1], should be the same value
        tot_num: the total number of spectrum for each mineral
        show: False
        norm_first: whether to perform normalisation before random interpolation or after
        norm_method: "max_1_min_0" or "max_1"
        sample_new: bool, if True, then generate new spectrum with randomly sampled coefficients. Otherwise, return
        the original ones
        lower_bound: int
        upper_bound: int
        sliding_window: int
        randshift: int
        train_or_test: str, "train", "test"
        aug_more_than_one: bool variable, whether apply noise/random shift on the minerals that have more than one
        spectra while sampling
        noise_std: the std for the gaussian noise
        mode: decides what kind of augmentation that I will use: repeat/pure_noise/the designed augmentation
    Ops:
        I will do this during the training process, in each epoch.
    """
    tr_class, freq = np.unique(label, return_counts=True)
    spectrum_update, real_or_sample, label_update = [], [], []
    if print_info:
        print("---------------------------------------------------------------")
        print("Sample %d spectra for per mineral" % tot_num)
        print("Add noise/randomshift the spectra for minerals that one more than one spectra", aug_more_than_one)
        if aug_more_than_one or np.min(freq) == 1:
            if train_or_test == "train":
                print("sliding window for calculating std", sliding_window)
                print("The std for the added gaussian noise", noise_std)
                if randshift != 0:
                    print("random shift %d between %d and %d" % (randshift, lower_bound, upper_bound))
        print("---------------------------------------------------------------")
    if len(np.shape(spectrum)) == 3:
        spectrum = spectrum[:, 1]

    for i, _class in enumerate(tr_class):
        sample = tot_num - freq[i]
        _index = np.where(label == _class)[0]
        select_spectrum = spectrum[_index]  # , 1, :]
        if norm_first:
            select_spectrum = np.array([norm(v, norm_method) for v in select_spectrum])
        if sample > 0 and sample_new is True:
            if freq[i] > 1 and train_or_test == "train":
                ratio = np.random.random([sample, freq[i]])
                ratio = ratio / np.sum(ratio, axis=-1, keepdims=True)
                sample = np.dot(ratio, select_spectrum)  # [num_samples, wavenumbers]
            elif freq[i] == 1 and train_or_test == "test":
                ratio = np.random.random([sample, freq[i]])
                ratio = ratio / np.sum(ratio, axis=-1, keepdims=True)
                sample = np.dot(ratio, select_spectrum)  # [generated number, wavelength]
            elif freq[i] == 1 and train_or_test == "train":
                sample = aug_mineral_npy(tot_num - freq[i], select_spectrum[0], sliding_window, randshift, lower_bound,
                                         upper_bound, mode=mode, norm_std=noise_std)
            spectrum_update.append(np.vstack([select_spectrum, sample]))
            real_or_sample.append(np.vstack([np.ones([freq[i], 1]), np.zeros([tot_num - freq[i], 1])]))
        else:
            spectrum_update.append(select_spectrum)
            real_or_sample.append(np.ones([freq[i], 1]))
        label_update.append(np.ones(len(spectrum_update[-1])) * _class)
    if not norm_first:
        if print_info:
            print("Normalizing my spectra to 0, 1")
            print("Normalization method", norm_method)
            print("The input spectrum for normalization shape", np.shape(spectrum_update[0][0]))
        spectrum_update = np.array([norm(v, norm_method) for j in spectrum_update for v in j])
    else:
        spectrum_update = np.array([v for j in spectrum_update for v in j])
    label_update = np.array([v for j in label_update for v in j])
    real_or_sample = np.array([v for j in real_or_sample for v in j])
    if show:
        fig = plt.figure(figsize=(10, 5))
        _label_select = np.random.choice(tr_class, 1)[0]
        _subset_index = np.where(label_update == _label_select)[0]
        for i in range(2):
            if i == 0:
                _value = spectrum_update[_subset_index]
            else:
                _value = [norm(v, "max_1_min_0") for v in spectrum_update[_subset_index]]
            ax = fig.add_subplot(2, 1, i + 1)
            _avg = np.mean(_value, axis=0)
            _std = 1.95 * np.std(_value) / np.sqrt(len(_value))
            ax.plot(np.arange(len(_avg)), _avg, 'r')
            ax.fill_between(np.arange(len(_avg)), _avg - _std, _avg + _std, color='r', alpha=0.4)
            ax.set_title(_label_select)
    return spectrum_update, label_update.astype('int32'), real_or_sample.astype('int32')


def save_spectrum(path, filenames):
    if "rs_dataset" in filenames[0]:
        spec = [get_spectrum(v) for v in filenames]
    else:
        spec = [get_spectrum(path + v) for v in filenames]
    print("There are %d spectra" % len(filenames))
    wavenumber = [v[:, 0] for v in spec]
    min_wave = np.array([np.min(v) for v in wavenumber])
    max_wave = np.array([np.max(v) for v in wavenumber])
    print("Min wavenumber:", np.min(min_wave), "Max wavenumber", np.max(max_wave))
    print(np.shape(spec))
    if "rs_dataset" in filenames[0]:
        pickle.dump(spec, open("../rs_dataset/raw_spectra.obj", "wb"))
    else:
        pickle.dump(spec, open(path + "/spectra.obj", "wb"), )


def get_interpolated_spectrum(min_wave, max_wave, spectra):
    target_wavenum = np.arange(max_wave)[min_wave:]
    spectrum_group = []
    for i in range(len(spectra)):
        single_spec = spectra[i]
        target_spectrum = find_interp(single_spec, target_wavenum)
        spectrum_group.append(np.vstack([[target_wavenum], [target_spectrum]]))
    spectrum_group = np.array(spectrum_group)
    return spectrum_group
