"""
Created on 11:47 at 09/07/2021
@author: bo 
"""
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import data.rruff as rruff
import data.organic as organic
import data.bacteria as bacteria


def read_dataset(raman_type, model_dir, min_wave, max_wave, tot_num_per_mineral,
                 norm_first=False, norm_method="max_1_min_0", lower_bound=10, upper_bound=1280,
                 sliding_window=10, randshift=3, noise_std=1, pre_define_tt_filenames=False, min_freq=2,
                 read_twin_triple="twin", beta=0.6, show=False,
                 random_leave_one_out=False, augmentation_mode_for_single_spectrum="history", print_info=True,
                 data_path="../"):
    """
    Args:
        raman_type: excellent_unoriented, raw, organic_target, organic_target_raw, bacteria
        model_dir: the directory to save the data, this is only used for the RRUFF dataset
        min_wave: the minimum wavenumber to cut the spectrum
        max_wave: the maximum wavenumber to cut the spectrum
        tot_num_per_mineral: the number of sampled spectra in the training dataset
        norm_first: bool False
        norm_method: "max_1_min_0", "none"
        lower_bound: If I am applying random shiftting on the spectra, then this argument determines the left limit
        upper_bound: similar to the lower_bound
        sliding_window: the window size that I used to calculate the standard deviation
        randshift: the shiftting number
        noise_std: the standard for the randomly sampled noise
        pre_define_tt_filenames: bool variable.
        min_freq: the minium frequency that I am using, default to be 2
        read_twin_triple: str, how to read the positive pair index and negative pair index
        beta: float number, determines the number of actual negative pairs that I am using
        show: bool
        random_leave_one_out: bool, this argument will be used when I do the experiment on the RRUFF dataset
        augmentation_mode_for_single_spectrum: str, "history", "pure_noise", "repeat"
    """
    if print_info:
        print("Read or save data from ", model_dir)
    filename_group = ["tr_filename", "tr_label", "tt_filename", "tt_label", "tr_index", "tt_index"]
    spectra_name_group = ["tr_spectra", "tr_label", "tt_spectra", "tt_label"]
    if raman_type == "excellent_unoriented":
        dir2read = data_path + "mineral_preprocess/"
        data_dir = dir2read + "excellent_unoriented_unoriented.csv"
        data = rruff.give_subset_of_spectrums(data_dir, None, "preprocess",
                                              print_info=print_info)
        tr_filename, tt_filename, \
            tr_label, tt_label, label_name, \
            tr_index, tt_index = rruff.mini_dataset(data, min_freq, 1, random_leave_one_out=random_leave_one_out,
                                                    print_info=print_info)
        spectra_path = dir2read + "spectra.obj"
        if pre_define_tt_filenames:
            if print_info:
                print("....I am loading the data from the model dir", model_dir)
            data_group = []
            for v in filename_group:
                if os.path.isfile(model_dir + v + ".npy"):
                    if "filename" in v:
                        _data = np.load(model_dir + v + ".npy", allow_pickle=True)
                    else:
                        _data = np.load(model_dir + v + ".npy")
                else:
                    _data = []
                data_group.append(_data)
            tr_filename, tr_label, tt_filename, tt_label, tr_index, tt_index = data_group

        if model_dir and not pre_define_tt_filenames:
            if print_info:
                print(".....I am saving the data to the model dir", model_dir)
            value_g = [tr_filename, tr_label, tt_filename, tt_label, tr_index, tt_index]
            for s_v, s_f in zip(value_g, filename_group):
                np.save(model_dir + "/%s" % s_f, s_v)
            np.save(model_dir + "/label_name", label_name)

    elif raman_type == "raw":
        data = rruff.give_all_raw(data_path + "/mineral_raw/", print_info=print_info)
        spectra_path = data_path + "/mineral_raw/spectra.obj"
        tr_filename, tt_filename, \
            tr_label, tt_label, label_name, \
            tr_index, tt_index = rruff.mini_dataset(data, 2, 1, random_leave_one_out=random_leave_one_out,
                                                    print_info=print_info)
        if model_dir and not pre_define_tt_filenames:
            value_g = [tr_filename, tr_label, tt_filename, tt_label, tr_index, tt_index]
            for s_v, s_f in zip(value_g, filename_group):
                np.save(model_dir + "/%s" % s_f, s_v)
            np.save(model_dir + "/label_name", label_name)
        if pre_define_tt_filenames:
            data_group = []
            for v in filename_group:
                if os.path.isfile(model_dir + v + ".npy"):
                    if "filename" in v:
                        _data = np.load(model_dir + v + ".npy", allow_pickle=True)
                    else:
                        _data = np.load(model_dir + v + ".npy")
                else:
                    _data = []
                data_group.append(_data)
            tr_filename, _, tt_filename, _, tr_index, tt_index = data_group

    elif raman_type == "organic_target" or raman_type == "organic_target_raw":
        preprocess = [True if "raw" not in raman_type else False][0]
        if pre_define_tt_filenames:
            if print_info:
                print("-----I am loading the data from--------", model_dir)
            data_group = []
            for v in spectra_name_group:
                _data = np.load(model_dir + v + ".npy")
                data_group.append(_data)
            tr_spectra, tr_label, tt_spectra, tt_label = data_group
        else:
            [tr_spectra, tr_label], \
                [tt_spectra, tt_label] = organic.get_target_data_with_randomleaveone(data_path+"organic", preprocess,
                                                                                     random_leave_one_out)
        if random_leave_one_out and not pre_define_tt_filenames:
            if print_info:
                print("saving data to", model_dir)
            value_g = [tr_spectra, tr_label, tt_spectra, tt_label]
            for s_v, s_f in zip(value_g, spectra_name_group):
                np.save(model_dir + "/%s" % s_f, s_v)
        tr_spectra = np.array([[np.arange(len(v)), v] for v in tr_spectra])
        tt_spectra = np.array([[np.arange(len(v)), v] for v in tt_spectra])
        label_name = np.array(["class-%d" % i for i in np.unique(tr_label)])

    elif "bacteria" in raman_type:
        reference_spectra, finetune_spectra, \
            test_spectra, label_name = bacteria.get_reference_data(data_path + "bacteria_reference_finetune/",
                                                                   False, None, False)
        label_name = np.array(label_name)
        if raman_type == "bacteria_reference_finetune":
            ref_spec, ref_label = reference_spectra
            fin_spec, fin_label = finetune_spectra
            tr_spectra, tr_label = [], []
            for i in np.unique(fin_label):
                _tr_ind = np.where(ref_label == i)[0]
                _fin_ind = np.where(fin_label == i)[0]
                tr_spectra.append(np.concatenate([ref_spec[_tr_ind[:100]],
                                                  fin_spec[_fin_ind]], axis=0))
                tr_label.append(np.repeat(i, 200))
            tr_spectra = np.array([v for j in tr_spectra for v in j])
            tr_label = np.concatenate(tr_label, axis=0)
            num_val = 20
        elif raman_type == "bacteria_random_reference_finetune":
            if pre_define_tt_filenames and model_dir:
                if print_info:
                    print("-----I am loading the saved training data from folder:", model_dir)
                tr_spectra = np.load(model_dir + "/tr_spectra.npy")
                tr_label = np.load(model_dir + "/tr_label.npy")
            else:
                ref_spec, ref_label = reference_spectra
                fin_spec, fin_label = finetune_spectra
                tr_spectra, tr_label = [], []
                selected_ref_index = []
                for i in np.unique(fin_label):
                    _tr_ind = np.where(ref_label == i)[0]
                    _fin_ind = np.where(fin_label == i)[0]
                    _rand_select_ind = np.random.choice(_tr_ind, 100, replace=False)
                    tr_spectra.append(np.concatenate([ref_spec[_rand_select_ind],
                                                      fin_spec[_fin_ind]], axis=0))
                    tr_label.append(np.repeat(i, 200))
                    selected_ref_index.append(_rand_select_ind)
                tr_spectra = np.array([v for j in tr_spectra for v in j])
                tr_label = np.concatenate(tr_label, axis=0)
            num_val = 20
            if not pre_define_tt_filenames:
                if model_dir:
                    if print_info:
                        print("----Save the data into folder", model_dir)
                    np.save(model_dir + "/selected_tr_index", np.array([v for j in selected_ref_index for v in j]))
                    np.save(model_dir + "/tr_spectra", tr_spectra)
                    np.save(model_dir + "/tr_label", tr_label)
        tt_spectra, tt_label = test_spectra
        if "bacteria" in raman_type:
            tr_index, val_index = [], []
            for i in np.unique(tr_label):
                _tr_index = np.where(tr_label == i)[0]
                if raman_type != "bacteria_reference_finetune" and raman_type != "bacteria_random_reference_finetune":
                    tr_index.append(_tr_index[:-num_val])
                    val_index.append(_tr_index[-num_val:])
                else:
                    _num_sample = len(_tr_index)
                    _tr_ref, \
                        _tr_fin = _tr_index[:_num_sample//2][:-num_val//2], _tr_index[_num_sample//2:][:-num_val//2]
                    _val_ref, \
                        _val_fin = _tr_index[:_num_sample//2][-num_val//2:], _tr_index[_num_sample//2:][-num_val//2:]
                    tr_index.append(np.concatenate([_tr_ref, _tr_fin], axis=0))
                    val_index.append(np.concatenate([_val_ref, _val_fin], axis=0))
            tr_index = np.array([v for j in tr_index for v in j])
            val_index = np.array([v for j in val_index for v in j])
            val_spectra, val_label = tr_spectra[val_index], tr_label[val_index]
            tr_spectra, tr_label = tr_spectra[tr_index], tr_label[tr_index]
        else:
            val_spectra, val_label = [], []
        if print_info:
            print("The shape of the validation dataset", np.shape(val_spectra), np.shape(val_label))
            print("The shape of the training dataset", np.shape(tr_spectra), np.shape(tr_label))

    if "bacteria" not in raman_type:
        val_spectra, val_label = [], []

    if "organic" not in raman_type and "bacteria" not in raman_type:
        if os.path.isfile(spectra_path):
            spectra = pickle.load(open(spectra_path, 'rb'))
            tr_spectra = [spectra[i] for i in tr_index]
            tt_spectra = [spectra[i] for i in tt_index]
        else:
            tr_spectra, tt_spectra = [], []

    tr_spectrum = rruff.get_interpolated_spectrum(min_wave, max_wave, tr_spectra)
    tt_spectrum = rruff.get_interpolated_spectrum(min_wave, max_wave, tt_spectra)
    if len(val_label) > 0:
        val_spectrum = rruff.get_interpolated_spectrum(min_wave, max_wave, val_spectra)
    sample_new = [True if tot_num_per_mineral > 0 else False][0]
    if not sample_new:
        if print_info:
            print("I am not sampling any new datapoints in this experiment")
        tr_spectrum = rruff.sample_test(tr_spectrum, norm_method)
        tt_spectrum = rruff.sample_test(tt_spectrum, norm_method)
        tr_real_or_sample = np.ones_like(tr_label)
        tt_real_or_sample = np.ones_like(tt_label)
        if len(val_label) > 0:
            val_spectrum = rruff.sample_test(val_spectrum, norm_method)
    else:
        tr_spectrum, tr_label, tr_real_or_sample = rruff.sample(tr_spectrum, tr_label, tot_num_per_mineral, show=False,
                                                                norm_first=norm_first, norm_method=norm_method,
                                                                sample_new=sample_new, lower_bound=lower_bound,
                                                                upper_bound=upper_bound, sliding_window=sliding_window,
                                                                randshift=randshift,
                                                                train_or_test="train",
                                                                aug_more_than_one=False,
                                                                noise_std=noise_std,
                                                                mode=augmentation_mode_for_single_spectrum)
        tt_spectrum, tt_label, tt_real_or_sample = rruff.sample(tt_spectrum, tt_label, 0, False, norm_first,
                                                                norm_method,
                                                                False, lower_bound, upper_bound, 0, 0, "test",
                                                                False, 0)
        if len(val_label) > 0:
            val_spectrum, val_label, val_real_or_sample = rruff.sample(val_spectrum, val_label, 0, False, norm_first,
                                                                       norm_method, False, lower_bound,
                                                                       upper_bound, 0, 0, "test", False, 0)
    if print_info:
        print("Training: There are %d spectra after sampling" % (len(tr_spectrum)))
        print("Training: These spectra belong to %d classes" % len(np.unique(tr_label)))
        print("Training: %d are sampled from random combinations and %d are real spectra" % (
            int(np.sum(1 - tr_real_or_sample)), int(np.sum(tr_real_or_sample))), "shape", np.shape(tr_real_or_sample))
        print("Training: The maximum value and minimum value in a spectrum", np.max(tr_spectrum), np.min(tr_spectrum))
        print("Testing: There are %d spectra after sampling" % (len(tt_spectrum)))
        print("Testing: These spectra belong to %d classes" % len(np.unique(tt_label)))
        print("Testing: %d are sampled from random combinations and %d are real spectrum" % (
            int(np.sum(1 - tt_real_or_sample)), int(np.sum(tt_real_or_sample))))
        print("Testing: The maximum value and minimum value in a spectrum", np.max(tt_spectrum), np.min(tt_spectrum))
    if read_twin_triple == "twin":
        positive_pair, negative_pair = rruff.find_pos_neg_pair(tr_label, tr_real_or_sample, beta)
        ratio = len(negative_pair) // len(positive_pair)
        if print_info:
            print("There are %d positive pairs and %d negative pairs" % (len(positive_pair), len(negative_pair)))
            print("The number of negative pair is %d times larger than the number of positive pair" % ratio)
        paired_index = [positive_pair, negative_pair]
    elif read_twin_triple == "cls":
        paired_index = [[] for _ in range(2)]
        if print_info:
            print("I am doing classification experiment, there I don't need to have the paired index")
    if len(val_label) == 0:
        return [tr_spectrum, tr_label], [[], []], [tt_spectrum, tt_label], paired_index, label_name
    else:
        return [tr_spectrum, tr_label], [val_spectrum, val_label], [tt_spectrum, tt_label], paired_index, label_name


def give_wavenumber(args, dir2load_data):
    if args["raman_type"] == "raw" or args["raman_type"] == "excellent_unoriented":
        wavenumber = np.arange(args["max_wave"])[args["min_wave"]:]
    elif "organic" in args["raman_type"]:
        wavenumber = np.linspace(106.62457839661, 3416.04065695651, args["max_wave"] - args["min_wave"])
    elif "bacteria" in args["raman_type"]:
        wavenumber = np.load(dir2load_data + "/bacteria_reference_finetune/wavenumbers.npy")
    return wavenumber


def get_fake_reference_and_test_data(tr_data, num_tt, data="BACTERIA", print_info=False):
    tr_spectra, tr_label = tr_data
    reference_index, test_index = [], []
    for i in np.unique(tr_label):
        _index = np.where(tr_label == i)[0]
        if data == "BACTERIA":
            reference_index.append(_index[:-num_tt])
            test_index.append(_index[-num_tt:])
        elif data == "RRUFF" or data == "ORGANIC":
            if len(_index) == 1:
                reference_index.append(_index)
                test_index.append(_index)
            else:
                reference_index.append(_index[:-num_tt])
                test_index.append(_index[-num_tt:])
    reference_index = np.array([v for j in reference_index for v in j])
    test_index = np.array([v for j in test_index for v in j])
    if print_info:
        print("There are %d reference spectra and %d test spectra" % (len(reference_index), len(test_index)))
    reference_data, reference_label = tr_spectra[reference_index], tr_label[reference_index]
    test_data, test_label = tr_spectra[test_index], tr_label[test_index]
    return [reference_data, reference_label], [test_data, test_label]


class MineralDataLoad(Dataset):
    def __init__(self, spectrum, label, index, transform):
        """Load the mineral dataset
        Args:
            spectrum: [number of spectrum, wavenumber]
            label: [number of spectrum]
            index: positive/negative pair index
            transform: the transformation
        """
        self.data = spectrum
        self.label = label
        self.index = index
        self.num_data = len(index)
        self.transform = transform
        print("There are %d spectra" % len(self.data))
        print("There are %d pairs" % len(self.index))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        _index = self.index[idx]
        _spectrum = self.data[_index[:2]]
        _sample = {"spectrum": _spectrum}
        if self.transform:
            _spectrum = self.transform(_sample)
        _la = _index[2:]
        return _spectrum["spectrum"], _la, 0


class MineralDataLoadHardManu(Dataset):
    def __init__(self, spectrum, label, index, transform, hard_negative_transform):
        """Load the mineral dataset with manually generated hard negative samples
        Args:
            spectrum: [number of spectrum, wavenumber]
            label: [number of spectrum]
            index: positive/negative pair index
            transform: the transformation
            hard_negative_transform: the transformation to generate hard negative samples
        """
        self.data = spectrum
        self.label = label
        self.index = index
        self.num_data = len(index)
        self.transform = transform
        self.hard_negative_transformation = hard_negative_transform
        print("There are %d spectra" % len(self.data))
        print("There are %d pairs" % len(self.index))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        _index = self.index[idx]
        if _index[-1] == 1:
            _spectrum = self.data[_index[:2]]
            _sample = {"spectrum": _spectrum, "shift": 0}
            if self.transform:
                _sample = self.transform(_sample)
                _spectrum = _sample["spectrum"]
        elif _index[-1] == 0:
            _spectrum = self.data[_index[:2]]
            _sample = {"spectrum": _spectrum, "shift": 0}
            if self.hard_negative_transformation:
                _sample = self.hard_negative_transformation(_sample)
                _spectrum = _sample["spectrum"]
        _move = _sample["shift"]
        _la = _index[2:]
        return _spectrum, _la, _move


class RandMove(object):
    def __init__(self, randshift, shift_std, shift_mean=30):
        """
        :param randshift: the scale to enlarge a Gaussian distribution
        :param shift_std: the std for the Gaussian distribution
        :param shift_mean: the mean for the Gaussian distribution
        Ops:
            larger randshift, shift_std, shift_mean are for hard-negative pairs
            smaller randshift, shift_std, shift_mean are for sampled positive pairs
        """
        self.randshift = randshift
        self.shift_std = shift_std
        self.shift_mean = shift_mean

    def __call__(self, sample):
        spectrum = sample["spectrum"]
        move = int(np.random.normal(self.shift_mean, self.shift_std, 1) * self.randshift)
        move = move * np.random.choice([1, -1], [1])[0]
        zeros = np.zeros([1, abs(move)])
        if move > 0:
            spectrum = np.concatenate([zeros, spectrum[:, :-move]], axis=1)
        elif move < 0:
            spectrum = np.concatenate([spectrum[:, abs(move):], zeros], axis=1)
        sample = {"spectrum": spectrum, "shift": move}
        return sample


class RandMoveTwin(object):
    def __init__(self, randshift, shift_std, shift_mean=30):
        """
        :param randshift: the scale to enlarge a Gaussian distribution
        :param shift_std: the std for the Gaussian distribution
        :param shift_mean: the mean for the Gaussian distribution
        Ops:
            larger randshift, shift_std, shift_mean are for hard-negative pairs
            smaller randshift, shift_std, shift_mean are for sampled positive pairs
        """
        self.randshift = randshift
        self.shift_std = shift_std
        self.shift_mean = shift_mean

    def __call__(self, sample):
        spectrum = sample["spectrum"]
        move = (np.random.normal(self.shift_mean, self.shift_std, [2]) * self.randshift).astype('int32')
        move = move * np.random.choice([1, -1], [2])
        for i, _move in enumerate(move):
            zeros = np.zeros([abs(_move)])
            if _move > 0:
                spectrum[i] = np.concatenate([zeros, spectrum[i][:-_move]])
            elif _move < 0:
                spectrum[i] = np.concatenate([spectrum[i][abs(_move):], zeros])
        sample = {"spectrum": spectrum, "shift": np.sum(abs(move))/2}
        return sample


class RandMoveTwinHard(object):
    def __init__(self, shift_scale, shift_std, shift_mean=30):
        """
        :param randshift: the scale to enlarge a Gaussian distribution
        :param shift_std: the std for the Gaussian distribution
        :param shift_mean: the mean for the Gaussian distribution
        Ops:
            larger randshift, shift_std, shift_mean are for hard-negative pairs
            smaller randshift, shift_std, shift_mean are for sampled positive pairs
        """
        self.shift_scale = shift_scale
        self.shift_std = shift_std
        self.shift_mean = shift_mean

    def __call__(self, sample):
        spectrum = sample["spectrum"]
        move = int(np.random.normal(self.shift_mean, self.shift_std, 1) * self.shift_scale)
        move = move * np.random.choice([1, -1], [1])[0]
        zeros = np.zeros([abs(move)])
        if move > 0:
            spectrum[1] = np.concatenate([zeros, spectrum[1][:-move]])
        elif move < 0:
            spectrum[1] = np.concatenate([spectrum[1][abs(move):], zeros])
        sample = {"spectrum": spectrum, "shift": move}
        return sample


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, sample):
        spectrum = torch.from_numpy(sample["spectrum"]).to(torch.float32)
        spectrum = spectrum.unsqueeze(0)
        sample["spectrum"] = spectrum
        return sample


class ToFloatTensorArray(object):
    def __call__(self, spectrum):
        spectrum = torch.from_numpy(spectrum).to(torch.float32)
        spectrum = spectrum.unsqueeze(1)
        return spectrum


def give_transformation(transformation_option, shift_params, read_twin_triple="twin"):
    """Give the possible transformation groups
    Args:
        transformation_option:
            "none" none of the transformation will be applied, the input will just be the original dataset
            "sample": interpolate between samples within the same class
            "random_shift": random shift the spectrum left to right, here the arguments for lower_upper_bound and
                rand_shift will need to be used
            "noise": here some noise will be added to the original data
            "sample_noise": sample first and then add noise
            "sample_random_shift": sample first and then random shift it to left/right
            "sample_noise_random_shift": apply all the transformations
        shift_params: [shift_scale, shift_std, shift_mean]
        read_twin_triple: "twin", "twin_hard"
    Return:
          transformation composition
    """
    trans_list = []
    if len(shift_params) > 0 and "shift" in transformation_option:
        if read_twin_triple == "twin_hard_without_label":
            trans_list.append(RandMove(shift_params[0],
                                       shift_params[1],
                                       shift_params[2]))
        elif read_twin_triple == "twin" or read_twin_triple == "twin_hard":
            trans_list.append(RandMoveTwin(shift_params[0],
                                           shift_params[1],
                                           shift_params[2]))
    trans_list.append(ToFloatTensor())
    print("------------------------The required transformation------------------------------")
    print(trans_list)
    print("---------------------------------------------------------------------------------")
    trans = transforms.Compose(trans_list)
    return trans


def load_test_data(spectrum, num_wavelength, device):
    spectrum_tensor = torch.from_numpy(spectrum).to(torch.float32).unsqueeze(1).view(
        len(spectrum), 1, num_wavelength).to(device)
    return spectrum_tensor


def give_neg_transformation(shift_params):
    """Give the negative transformation"""
    shift_scale, shift_std, shift_mean = shift_params
    trans_list = []
    trans_list.append(RandMoveTwinHard(shift_scale, shift_std, shift_mean))
    trans_list.append(ToFloatTensor())
    trans = transforms.Compose(trans_list)
    return trans
