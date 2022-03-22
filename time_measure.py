import torch
import os
import numpy as np
import const
import data.prepare_data as pdd
import test as test
import vis_utils as vu
import paper_figures as pf
from scipy.special import softmax
import matplotlib.pyplot as plt
import time

dir2read_exp = "../exp_data/exp_group/"  # USER DEFINE
dir2read_data = "../data_group/"  # USER DEFINE


def measure_time(raman_type):
    model_dir = dir2read_exp + raman_type + "/" 
    model_dir = [model_dir + v for v in os.listdir(model_dir) if "version_" in v][0]
    dir2load_data = model_dir + "/data_splitting/"
    ckpt_dir = [model_dir + "/" + v for v in sorted(os.listdir(model_dir)) if "repeat_" in v]

    args = const.give_args_test(raman_type=raman_type)
    args["pre_define_tt_filenames"] = True
    if "bacteria" in raman_type:
        tr_data, val_data, tt_data, _, label_name = test.get_data(args, dir2load_data, "cls", False, dir2read_data)
    else:
        tr_data, tt_data, _, label_name = test.get_data(args, dir2load_data, "cls", False, dir2read_data)
        val_data = []
    string_use = ["ensemble_%d" % i for i in range(len(ckpt_dir))]
    prediction_g = {}
    time_group = []
    for i, s_ckpt in enumerate(ckpt_dir):
        prediction, _, _time = test.get_model_baseon_modeldir(args, s_ckpt, 
                                                    [tr_data, tt_data, label_name],
                                                    print_info=True)
        time_group.append(_time)
        prediction_g[string_use[i]] = prediction
    group_index = [np.where(tr_data[1] == i)[0] for i in np.unique(tt_data[1])]
    prediction_g, string_use = test.add_ensemble(prediction_g, string_use, False, group_index, tr_data[1], tt_data[1])
    print("================================================================================")
    print("There are %d test spectra and %d reference spectra" % (len(tt_data[0]), len(tr_data[0])))
    print("The average time per spectrum", np.mean(time_group) / len(tt_data[0]))
    