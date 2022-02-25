"""
Created on 10:25 at 08/07/2021/
@author: bo
"""
import argparse
import os
import numpy as np
import pickle
import data.rruff as rruff
from sklearn.metrics import roc_curve, auc
from scipy.special import expit, softmax
import const
import test
import vis_utils as vis_utils
import data.prepare_data as pdd
import matplotlib
import matplotlib.ticker as ticker
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'text.usetex': True,
# })
matplotlib.rcParams.update({
    'font.family': 'serif',
    "font.size": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.title_fontsize": 7,
    "axes.titlesize": 7,
})
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter, NullFormatter
import matplotlib.ticker as mticker

TEXTWIDTH = 6.75133


def give_args():
    """This function is used to give the argument"""
    parser = argparse.ArgumentParser(description='Reproduce figures in the paper')
    parser.add_argument('--dir2read_exp', type=str, default="../exp_data/exp_group/")
    parser.add_argument('--dir2read_data', type=str, default="../data_group/")
    parser.add_argument('--dir2save', type=str, default="figures/")
    parser.add_argument('--index', type=str, default="figure_1", help="which figure or table do you want to produce?")
    parser.add_argument("--save", type=const.str2bool, default=False, help="whether to save the image or not")
    parser.add_argument("--pdf_pgf", type=str, default="pgf", help="in what kind of format will I save the image?")
    return parser.parse_args()


# ------------------------------------------------------------------------------------


def set_size(width, fraction=1, enlarge=0):
    """
    Args:
        width: inches
        fraction: float
    """
    # Width of figure (in pts)
    fig_width_in = width * fraction
    golden_ratio = (5 ** .5 - 1) / 2
    if enlarge != 0:
        golden_ratio *= enlarge
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def give_figure_specify_size(fraction, enlarge=0):
    fig = plt.figure()
    fig.set_size_inches(set_size(TEXTWIDTH, fraction, enlarge))
    return fig


# -------------- First figure --------------------#

def give_data_augmentation_example(tds_dir_use="../exp_data/eerst_paper_figures/",
                                   save=False, pdf_pgf="pgf", data_path="../data_group/"):
    args = const.give_args_test(raman_type="excellent_unoriented")
    args["pre_define_tt_filenames"] = False
    tr_data, _, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls", dir2read=data_path)
    show_data_augmentation_example(args, tr_data[0], tr_data[1], label_name_tr,
                                   tds_dir_use, save, pdf_pgf)


def show_data_augmentation_example(args, tr_spectrum, tr_label, label_name_tr,
                                   tds_dir_use="../exp_data/eerst_paper_figures/",
                                   save=False, pdf_pgf="pdf"):
    """Illustrate the data augmentation process
    Args:
        args: the arguments that can tell me the maximum and minimum wavenumber
        tr_spectrum: [num_spectra, wavenumbers]
        tr_label: [num_spectra]
        label_name_tr: corresponding names for each class in the tr label
        tds_dir_use: the directory to save the data.
        save: bool, whether to save the figure
    """
    select_index = np.where(label_name_tr == "AlumNa")[0] #AlumNa
    tr_select = tr_spectrum[np.where(tr_label == select_index)[0]]
    u_spectrum = tr_select[np.random.choice(len(tr_select), 1)[0]]
    std_s_spectrum = rruff.calc_std(u_spectrum, 10)
    rand_noise = np.random.normal(0, 3, [3, len(u_spectrum)]) # 5 before
    generate = abs(np.expand_dims(u_spectrum, 0) + rand_noise * np.expand_dims(std_s_spectrum, 0))
    generate = generate / np.max(generate, axis=-1, keepdims=True)
    wavenumber = np.arange(args["max_wave"])[args["min_wave"]:]
    text_use = ["%s" % label_name_tr[select_index][0], "Synthetic"]
    fig = give_figure_specify_size(0.5, 1.1)
    ax = fig.add_subplot(111)
    for i, s_c in enumerate(["r", "g"]):
        ax.plot([], [], color=s_c)
    ax.plot(wavenumber, u_spectrum, 'r', lw=0.8)
    ax.text(250, 0.5, text_use[0])
    for i, s in enumerate(generate):
        ax.plot(wavenumber, s + i + 1, 'g', lw=0.8)
        ax.text(250, 0.5 + i + 1, text_use[-1])
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xlabel("Wavenumber (cm" + r"$^{-1})$")
    ax.set_ylabel("Intensity (a.u.)")
    if save:
        plt.savefig(
            tds_dir_use + "/augmentation_example_on_RRUFF_%s.%s" % (label_name_tr[select_index][0],
                                                                    pdf_pgf),
            pad_inches=0, bbox_inches='tight')


# --------------------------- second & third figure ------------------------------#


def show_example_spectra(tds_dir="../exp_data/eerst_paper_figures/", save=False, pdf_pgf="pgf",
                         data_path="../data_group/"):
    """This function shows the example spectra from each dataset. It should also show the distribution of the classes
    """
    dataset = ["RRUFF", "RRUFF", "ORGANIC", "ORGANIC", "BACTERIA"]
    raman_type = ["raw", "excellent_unoriented", "organic_target_raw", "organic_target", "bacteria_reference_finetune"]
    color_group = ['r', 'g']
    fig = give_figure_specify_size(0.5, 3.0)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    im_index = 0
    title_group = ["Mineral (r)", "Mineral (p)", "Organic (r)", "Organic (p)", "Bacteria"]
    tr_frequency_count = []
    for s_data, s_raman in zip(dataset, raman_type):
        ax = fig.add_subplot(5, 1, im_index + 1)
        args = const.give_args_test(raman_type=s_raman)
        args["pre_define_tt_filenames"] = False
        if s_data == "RRUFF" or s_data == "ORGANIC":
            tr_data, _, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls", dir2read=data_path)
        else:
            tr_data, _, _, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls", dir2read=data_path)
        tr_spectra, tr_label = tr_data
        unique_label, unique_count = np.unique(tr_label, return_counts=True)
        if s_data == "RRUFF":
            tr_frequency_count.append(unique_count)
        if s_data == "RRUFF":
            class_name = "Beryl"
            select_label = np.where(label_name_tr == class_name)[0]
            index = np.where(tr_label == select_label)[0]
        else:
            select_label = unique_label[np.argmax(unique_count)]
            if s_data == "ORGANIC":
                select_label = 1
            class_name = label_name_tr[select_label]
            if s_data == "ORGANIC":
                class_name = "Benzidine"
            index = np.where(tr_label == select_label)[0]
        if len(index) > 15:
            index = np.random.choice(index, 5, replace=False)
        _spectra = tr_spectra[index]
        if s_data == "RRUFF":
            wavenumber = np.arange(args["max_wave"])[args["min_wave"]:]
            ax.set_xlim((0, 1500))
        elif s_data == "BACTERIA":
            wavenumber = np.load("../bacteria/wavenumbers.npy")
        elif s_data == "ORGANIC":
            wavenumber = np.linspace(106.62457839661, 3416.04065695651, np.shape(tr_spectra)[1])
        for j, s in enumerate(_spectra):
            ax.plot(wavenumber, s, alpha=0.8, lw=0.8)
        ax.set_title(title_group[im_index] + ": " + class_name)
        im_index += 1
        if s_raman == "bacteria_finetune":
            ax.set_xlabel("Wavenumber (cm" + r"$^{-1})$")
    ax_global.set_ylabel("Intensity (a.u.)\n\n")
    plt.subplots_adjust(hspace=0.47)
    if save:
        plt.savefig(tds_dir + "/example_spectra.%s" % pdf_pgf, pad_inches=0, bbox_inches='tight')
    title_group = ["Mineral (r)", "Mineral (p)"]
    fig = give_figure_specify_size(0.5, 0.8)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    max_count = np.max([np.max(np.unique(v, return_counts=True)[1]) for v in tr_frequency_count])
    for i, s in enumerate(tr_frequency_count):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(s, bins=np.max(s), ec="white", lw=0.4)
        ax.set_yscale("symlog")
        ax.set_ylim((0, max_count))
        if i == 1:
            ax.yaxis.set_ticks_position('none')
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.set_title(title_group[i])
    plt.subplots_adjust(wspace=0.04)
    ax_global.set_xlabel("\n\n Number of spectra per class")
    ax_global.set_ylabel("Number of classes \n\n")
    if save:
        plt.savefig(tds_dir + "/class_distribution_on_RRUFF.%s" % pdf_pgf, pad_inches=0, bbox_inches='tight')


# -------------------- figure 4 --------------------------


def give_uncertainty_distribution_figure_with_confidence_interval(tds_dir="../exp_data/eerst_paper_figures/",
                                                                  save=False,
                                                                  pdf_pgf="pgf",
                                                                  path_init="../", use_nll_or_prob="prob",
                                                                  data_path="../data_group/", strategy="sigmoid"):
    _, rruff_raw_avg, rruff_raw_std = get_multiple_rruff_uncertainty("raw", path_init,
                                                                     use_nll_or_prob=use_nll_or_prob,
                                                                     data_path=data_path, strategy=strategy)
    _, rruff_pre_avg, rruff_pre_std = get_multiple_rruff_uncertainty("excellent_unoriented", path_init,
                                                                     use_nll_or_prob=use_nll_or_prob,
                                                                     data_path=data_path, strategy=strategy)
    _, organic_raw_avg, organic_raw_std = get_multiple_organic_uncertainty("organic_target_raw", data_path=data_path, path_init=path_init, 
                                                                           use_nll_or_prob="prob", strategy=strategy)
    _, organic_pre_avg, organic_pre_std = get_multiple_organic_uncertainty("organic_target", data_path=data_path, path_init=path_init, 
                                                                           use_nll_or_prob="prob", strategy=strategy)
    _, bacteria_avg, bacteria_std = get_multiple_bacteria_uncertainty(path_init,
                                                                      use_nll_or_prob=use_nll_or_prob,
                                                                     data_path=data_path, strategy=strategy)
    color_use = ["r", "g", "b", "orange", "m"]
    title_group = "Correct match (%)"
    dataset = ["Mineral (r)", "Mineral (p)", "Organic (r)", "Organic (p)", "Bacteria"]
    fig = give_figure_specify_size(0.5, 1.25)
    ax = fig.add_subplot(111)
    for j, stat in enumerate([[rruff_raw_avg, rruff_raw_std],
                              [rruff_pre_avg, rruff_pre_std],
                              [organic_raw_avg, organic_raw_std],
                              [organic_pre_avg, organic_pre_std],
                              [bacteria_avg, bacteria_std]]):
        if strategy != "none":
            plot_fillx_filly(stat[0][0]*100, stat[1][0],
                            stat[0][1]*100, stat[1][1], ax, color_use=color_use[j])
        else:
            plot_fillx_filly(stat[0][0], stat[1][0], stat[0][1]*100, stat[1][1],
                             ax, color_use=color_use[j])
    ax.legend(dataset, loc='best', handlelength=1.1, handletextpad=0.5,
              borderpad=0.25)  # bbox_to_anchor=(1.0, 0.8), loc="upper left",
    if strategy == "softmax" or strategy == "sigmoid":
        ax.plot([0, 100], [0, 100], ls='--', color='black')
        ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))
    ax.set_ylabel(title_group)
    ax.yaxis.set_major_formatter(FuncFormatter(form3))
    ax.set_xlabel("Similarity score")
    if save:
        plt.savefig(tds_dir + "/uncertainty_distribution_for_the_test_dataset_with_confidence_interval_%s.%s" % (strategy, pdf_pgf),
                    pad_inches=0, bbox_inches='tight')
        
        
def motivation_for_conformal_prediction_bacteria(save=False, pdf_pgf="pgf",
                                                          path_init="../exp_data/exp_group/",
                                                          path2save="../exp_data/eerst_paper_figures/",
                                                          data_path="../data_group/"):
    dataset = ["BACTERIA"]
    output_bacteria = motivation_for_conformal_prediction(dataset[0], select_length=1, show=False, path_init=path_init,
                                                         data_path=data_path)
    two_select_index = np.where(np.array([len(v) for v in output_bacteria[4]]) == 2)[0]

    fig = give_figure_specify_size(1.1, 0.8)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])

    ax = fig.add_subplot(2, 2, 1)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=579, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    ax = fig.add_subplot(2, 2, 3)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=two_select_index[5], ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)

    ax = fig.add_subplot(1, 2, 2)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=463, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    plt.subplots_adjust(wspace=0.04)
    ax_global.set_xlabel("\nWavenumber (cm" + r"$^{-1}$" + ")")
    ax_global.set_ylabel("Intensity (a.u.) \n")


    return output_bacteria
    
    

    


def motivation_for_conformal_prediction_multiple_datasets(save=False, pdf_pgf="pgf",
                                                          path_init="../exp_data/exp_group/",
                                                          path2save="../exp_data/eerst_paper_figures/",
                                                          data_path="../data_group/"):
    dataset = ["RRUFF_excellent_unoriented",
               "RRUFF_raw",
               "BACTERIA"]
    fig = give_figure_specify_size(1.1, 0.8)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    output_rruff_r = motivation_for_conformal_prediction(dataset[1], select_length=1, show=False, path_init=path_init,
                                                         data_path=data_path)
    output_rruff_p = motivation_for_conformal_prediction(dataset[0], select_length=1, show=False, path_init=path_init,
                                                         data_path=data_path)
    output_bacteria = motivation_for_conformal_prediction(dataset[2], select_length=1, show=False, path_init=path_init,
                                                         data_path=data_path)
    ax = fig.add_subplot(2, 3, 1)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=579, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    ax = fig.add_subplot(2, 3, 4)
    _show_motivation_for_conformal_prediction(*output_rruff_p, select_index=25, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    ax = fig.add_subplot(1, 3, 2)
    _show_motivation_for_conformal_prediction(*output_rruff_r, select_index=145, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    ax = fig.add_subplot(1, 3, 3)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=463, ax=ax, save=False, pdf_pgf="None",
                                              path2save=path2save)
    plt.subplots_adjust(wspace=0.04)
    ax_global.set_xlabel("\nWavenumber (cm" + r"$^{-1}$" + ")")
    ax_global.set_ylabel("Intensity (a.u.) \n")
    if save:
        plt.savefig(path2save + "conformal_motivation.%s" % pdf_pgf, pad_inches=0, bbox_inches='tight')


def _calc_motivation_for_conformal_prediction(alpha_use=0.05, use_original_weight="original",
                                              dataset="BACTERIA",
                                              path_init="../exp_data/exp_group/",
                                              data_path="../data_group/"):
    if dataset == "BACTERIA":
        wavenumbers = np.load("../bacteria/wavenumbers.npy")
        raman_type = "bacteria_random_reference_finetune"
        args = const.give_args_test(raman_type=raman_type)
        args["pre_define_tt_filenames"] = False
        tr_data, val_data, tt_data, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls",
                                                                     print_info=False, dir2read=data_path)
        tr_spectra, tt_spectra = tr_data[0], tt_data[0]
        tr_label_group = [tr_data[1], tr_data[1]]
        val_label, tt_label = val_data[1], tt_data[1]
        path2load = path_init + "bacteria_reference_finetune/tds/"
        s_split = 1
        path = path2load + [v for v in os.listdir(path2load) if "split_%d" % s_split in v and ".txt" not in v][0] + "/"
        val_prediction = pickle.load(open(path + "validation_prediction.obj", "rb"))
        tt_prediction = pickle.load(open(path + "test_prediction.obj", "rb"))
    elif "RRUFF" in dataset:
        raman_type = dataset.split("RRUFF_")[1]
        dataset = "RRUFF"
        args = const.give_args_test(raman_type=raman_type)
        wavenumbers = np.arange(args["max_wave"])[args["min_wave"]:]
        args["pre_define_tt_filenames"] = False
        tr_data, tt_data, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls",
                                                           print_info=False, dir2read=data_path)
        [_, reference_val_label], [_, val_label] = pdd.get_fake_reference_and_test_data(tr_data, 1, data=dataset)
        tr_label_group = [reference_val_label, tr_data[1]]
        tr_spectra, tt_spectra = tr_data[0], tt_data[0]
        tt_label = tt_data[1]
        path2load = path_init + "%s/tds/" % raman_type
        s_split = 1
        path = path2load + [v for v in os.listdir(path2load) if "split_%d" % s_split in v and '.txt' not in v][0] + "/"
        val_prediction = pickle.load(open(path + "validation_prediction.obj", "rb"))
        tt_prediction = pickle.load(open(path + "test_prediction.obj", "rb"))
    if use_original_weight == "original":
        val_pred_en, tt_pred_en = val_prediction[0]["ensemble_avg"], tt_prediction[0]["ensemble_avg"]
    else:
        val_pred_en, tt_pred_en = val_prediction[1]["ensemble_avg"], tt_prediction[1]["ensemble_avg"]
    val_pred_baseon_cls, _ = test.reorganize_similarity_score(val_pred_en, tr_label_group[0])
    tt_pred_baseon_cls, tt_corr_tr_index = test.reorganize_similarity_score(tt_pred_en, tr_label_group[1])
    val_prediction_score = give_calibration_single_score_prediction(val_pred_baseon_cls, True, val_label)
    threshold = np.quantile(val_prediction_score, alpha_use)
    tt_top1 = np.argmax(tt_pred_baseon_cls, axis=-1)
    accu = [v == q for v, q in zip(tt_top1, tt_label)]
    tt_prediction, \
    tt_accuracy = give_test_prediction_baseon_single_score_threshold(tt_pred_baseon_cls,
                                                                     True, tt_label,
                                                                     threshold)
    tt_pred_softmax = softmax(tt_pred_baseon_cls, axis=-1)
    tt_correct_or_wrong = [1 if tt_label[i] in v else 0 for i, v in enumerate(tt_prediction)]
    return tr_label_group, [val_label, tt_label], [tr_spectra, tt_spectra], \
           tt_pred_softmax, tt_prediction, tt_correct_or_wrong, tt_corr_tr_index, label_name_tr, wavenumbers


def _show_motivation_for_conformal_prediction(tr_label_group, tt_label,
                                              tr_spectra, tt_spectra,
                                              tt_prediction, tt_pred_baseon_cls_softmax,
                                              tt_corr_tr_index,
                                              label_name,
                                              wavenumbers, select_index, ax, save, pdf_pgf, path2save):
    """Args
    select_index: a single index
    save: bool variable
    """
    _tr_corr_index = np.where(tr_label_group[1] == tt_label[select_index])[0]
    if len(tt_prediction[select_index]) >= 3:
        height = 1.5
    elif len(tt_prediction[select_index]) == 2:
        height = 1.2
    else:
        height = 1.0
    if not ax:
        fig = give_figure_specify_size(0.5, height)
        ax = fig.add_subplot(111)
    color_input = 'r'
    color_group = ['g', 'b', 'orange', "c", "tab:blue"]
    select_prediction = tt_prediction[select_index]
    score = tt_pred_baseon_cls_softmax[select_index]
    score_select = score[select_prediction]
    score_select_sort_index = np.argsort(score_select)[::-1]
    select_prediction = select_prediction[score_select_sort_index]
    score_select_sorted = score_select[score_select_sort_index]
    input_name = "Input:   %s" % label_name[tt_label[select_index]]
    scale = 1.4
    ax.plot(wavenumbers, tt_spectra[select_index] + len(select_prediction) * scale, color=color_input)
    if len(label_name) == 30:
        x_loc = 450
    else:
        x_loc = 100
    ax.text(x_loc, len(select_prediction) * scale + 0.95, input_name, color=color_input)
    for i, s in enumerate(select_prediction):
        if s == tt_label[select_index]:
            color_use = color_input
        else:
            color_use = color_group[i]
        _tr_corr_index = tt_corr_tr_index[select_index][s]
        match_name = "Match: %s (p=%.2f)" % (label_name[s], score_select_sorted[i])
        ax.plot(wavenumbers, tr_spectra[_tr_corr_index] + (len(select_prediction) - i - 1) * scale,
                color=color_use)
        ax.text(x_loc, (len(select_prediction) - i - 1) * scale + 1, match_name, color=color_use)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    if save:
        _name = label_name[tt_label[select_index]]
        plt.savefig(path2save + "conformal_motivation_%s_%d.%s" % (_name, select_index, pdf_pgf),
                    pad_inches=0, bbox_inches='tight')


def motivation_for_conformal_prediction(dataset="RRUFF_excellent_unoriented",
                                        select_length=3, path_init="../", show=False, save=False,
                                        pdf_pgf="pgf", data_path="../data_group/"):
    if dataset == "RRUFF_excellent_unoriented":
        alpha_use = 0.01
    elif dataset == "RRUFF_raw":
        alpha_use = 0.0005
    elif dataset == "BACTERIA":
        alpha_use = 0.05
    tr_label_group, [val_label, tt_label], [tr_spectra, tt_spectra], \
    tt_pred_softmax, tt_prediction, tt_correct_or_wrong, \
    tt_corr_tr_index, label_name, wavenumbers = _calc_motivation_for_conformal_prediction(alpha_use=alpha_use,
                                                                                          dataset=dataset,
                                                                                          path_init=path_init,
                                                                                          data_path=data_path)

    def filter_index(select_length):
        tt_index = []
        for i, v in enumerate(tt_prediction):
            prob_subset = tt_pred_softmax[i, v]
            prob_subset_sort_index = np.argsort(prob_subset)[::-1]
            _pred_label = np.array(v)[prob_subset_sort_index]
            if len(v) == select_length and tt_correct_or_wrong[i] == 1 and _pred_label[-1] == tt_label[i]:
                tt_index.append(i)
        return tt_index

    if select_length != 0:
        tt_index = filter_index(select_length)
        select_index = np.random.choice(tt_index, 1)
    else:
        if dataset == "RRUFF_raw":
            select_index = [191, 182, 145]
        elif dataset == "RRUFF_excellent_unoriented":
            select_index = [25, 594, 312, 1213, 53]
        elif dataset == "BACTERIA":
            select_index = [463]
    if show:
        for _select_index in select_index:
            _show_motivation_for_conformal_prediction(tr_label_group, tt_label,
                                                      tr_spectra, tt_spectra,
                                                      tt_prediction, tt_pred_softmax,
                                                      tt_corr_tr_index,
                                                      label_name, wavenumbers, _select_index, ax=None, save=save,
                                                      pdf_pgf=pdf_pgf, path2save=None)
    return tr_label_group, tt_label, tr_spectra, tt_spectra, tt_prediction, tt_pred_softmax, tt_corr_tr_index, \
           label_name, wavenumbers
           
           
def give_conformal_prediction_for_bacteria_paper(path_init="../",
                                                use_original_weight="original",
                                                tds_dir=None, save=False, pdf_pgf="pdf",
                                                data_path="../data_group/", 
                                                apply_softmax="none"):
    
    alpha_group = np.linspace(0, 0.20, 10)
    path2load, split_version = get_path_for_conformal(path_init, "bacteria_reference_finetune")
    stat_bacteria = main_plot_for_scoring_rule(path2load, split_version,
                                               "bacteria_random_reference_finetune",
                                               "BACTERIA", use_original_weight,
                                               alpha_group, show=False, data_path=data_path, apply_softmax=apply_softmax)
    
    fig = give_figure_specify_size(1.0, 0)
    
    title_group = ["Bacteria: 82.71"]
    loc = [[0.80, 0.92]]
    orig_perf = [82.71]
    orig_perf = [v - 1 for v in orig_perf]
    
    
    for i, stat in enumerate([stat_bacteria]):
        stat_avg = np.mean(stat, axis=0)
        ax = fig.add_subplot(2, 2, 1)
        x_axis = 100 - alpha_group * 100 
        ax.plot(x_axis, stat_avg[:, 0] * 100, color='r', marker='.')
        ax.plot(x_axis, x_axis, color='g', ls=':')
        ax.yaxis.set_major_formatter(FuncFormatter(form3))
        ax.set_xlim(np.min(x_axis), np.max(x_axis))
        ax.set_ylim(np.min(x_axis), np.max(x_axis))
        ax.set_ylabel("Empirical coverage (%)")
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        # plt.axis('square')
        ax.set_title(title_group[i])
        
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(x_axis, stat_avg[:, 1], color='b', marker='.')
        # plt.axis('square')
        # ax.set_yscale("symlog")

        ax.set_ylabel("Average set size")
        ax.set_xlabel("Theoretical coverage (1 - " + r'$\alpha$' + ")" + "(%)")
        
        ax.yaxis.set_major_formatter(FuncFormatter(form3))
        ax.xaxis.set_major_formatter(FuncFormatter(form3))    
    
    dataset = ["BACTERIA"]
    output_bacteria = motivation_for_conformal_prediction(dataset[0], select_length=1, show=False, path_init=path_init,
                                                         data_path=data_path)
    two_select_index = np.where(np.array([len(v) for v in output_bacteria[4]]) == 2)[0]

    # fig = give_figure_specify_size(1.1, 0.8)
    # ax_global = vis_utils.ax_global_get(fig)
    # ax_global.set_xticks([])
    # ax_global.set_yticks([])

    # ax = fig.add_subplot(3, 2, 2)
    # _show_motivation_for_conformal_prediction(*output_bacteria, select_index=579, ax=ax, save=False, pdf_pgf="None",
    #                                           path2save=None)
    # ax.xaxis.set_major_formatter(plt.NullFormatter())

    
    ax = fig.add_subplot(2, 2, 2)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=two_select_index[-4], ax=ax, save=False, pdf_pgf="None",
                                              path2save=None)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_title("Example prediction set")
    ax.set_ylabel("Intensity (a.u.)")


    ax = fig.add_subplot(2, 2, 4)
    _show_motivation_for_conformal_prediction(*output_bacteria, select_index=463, ax=ax, save=False, pdf_pgf="None",
                                              path2save=None)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlabel("Wavenumber")
    # plt.subplots_adjust(wspace=0.23)
    # ax_global.set_xlabel("\nWavenumber (cm" + r"$^{-1}$" + ")")
    # ax_global.set_ylabel("Intensity (a.u.) \n")
    
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    if save:
        if pdf_pgf == "pdf":
            plt.savefig(tds_dir + "/correlation_between_alpha_and_accuracy_and_set_size_%s.pdf" % apply_softmax,
                        pad_inches=0, bbox_inches='tight')
        elif pdf_pgf == "pgf":
            plt.savefig(tds_dir + "/correlation_between_alpha_and_accuracy_and_set_size.pgf",
                        pad_inches=0, bbox_inches='tight')



def give_conformal_prediction_for_multiple_datasets(path_init="../",
                                                    use_original_weight="weighted",
                                                    tds_dir=None, save=False, pdf_pgf="pdf",
                                                    data_path="../data_group/"):
    # rruff raw
    alpha_group_group = []
    alpha_group = np.linspace(0, 0.03, 10)
    alpha_group_group.append(alpha_group)
    path2load, split_version = get_path_for_conformal(path_init, "raw")
    stat_rruff_raw = main_plot_for_scoring_rule(path2load, split_version,
                                                "raw", "RRUFF", use_original_weight,
                                                alpha_group, show=False, data_path=data_path)

    alpha_group = np.linspace(0, 0.05, 10)
    alpha_group_group.append(alpha_group)
    path2load, split_version = get_path_for_conformal(path_init, "excellent_unoriented")
    stat_rruff_preprocess = main_plot_for_scoring_rule(path2load, split_version,
                                                       "excellent_unoriented", "RRUFF",
                                                       "original", alpha_group, show=False, data_path=data_path)

    alpha_group = np.linspace(0, 0.011, 10)
    alpha_group_group.append(alpha_group)
    path2load, split_version = get_path_for_conformal(path_init, "organic_target_raw")
    stat_organic_raw = main_plot_for_scoring_rule(path2load, split_version, "organic_target_raw", "ORGANIC",
                                                  "original", alpha_group, show=False, data_path=data_path)

    alpha_group = np.linspace(0, 0.04, 10)
    alpha_group_group.append(alpha_group)
    path2load, split_version = get_path_for_conformal(path_init, "organic_target")
    stat_organic = main_plot_for_scoring_rule(path2load, split_version, "organic_target", "ORGANIC",
                                              "original", alpha_group, show=False, data_path=data_path)

    alpha_group = np.linspace(0, 0.20, 10)
    alpha_group_group.append(alpha_group)
    path2load, split_version = get_path_for_conformal(path_init, "bacteria_reference_finetune")
    stat_bacteria = main_plot_for_scoring_rule(path2load, split_version,
                                               "bacteria_random_reference_finetune",
                                               "BACTERIA", use_original_weight,
                                               alpha_group, show=False, data_path=data_path)

    fig = give_figure_specify_size(0.5, 4.0)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_global.spines['top'].set_visible(False)
    ax_global.spines['right'].set_visible(False)
    ax_global.spines['bottom'].set_visible(False)
    ax_global.spines['left'].set_visible(False)

    title_group = ["Mineral (r): 94.48", "Mineral (p): 91.86", "Organic (r): 98.26", "Organic (p): 98.26",
                   "Bacteria: 82.71"]
    loc = [[0.97, 0.958], [0.95, 0.95], [0.989, 0.987], [0.96, 0.987], [0.80, 0.92]]
    orig_perf = [94.48, 91.86, 98.26, 98.26, 82.71]
    orig_perf = [v - 1 for v in orig_perf]
    for i, stat in enumerate([stat_rruff_raw, stat_rruff_preprocess,
                              stat_organic_raw, stat_organic, stat_bacteria]):
        stat_avg = np.mean(stat, axis=0)
        ax = fig.add_subplot(len(title_group), 1, i + 1)
        vis_utils.show_twinx(alpha_group_group[i] * 100, stat_avg[:, 0] * 100, stat_avg[:, 1],
                             ax=ax)
        ax.set_title(title_group[i])
        ax.set_ylim(bottom=orig_perf[i])
        ax.set_yticks(np.linspace(orig_perf[i], 100, 4))
        ax.yaxis.set_major_formatter(FuncFormatter(form3))
        ax.xaxis.set_major_formatter(FuncFormatter(form3))

    ax_global.set_ylabel("Empirical coverage (%) \n\n\n", color='r')
    ax_global_t = ax_global.twinx()
    ax_global_t.set_yticks([])
    ax_global_t.spines['top'].set_visible(False)
    ax_global_t.spines['right'].set_visible(False)
    ax_global_t.spines['bottom'].set_visible(False)
    ax_global_t.spines['left'].set_visible(False)
    # ax_global_t.grid(None)
    ax_global_t.set_ylabel("\n\n\n Average set size", color='g')
    ax_global.set_xlabel("\n \n Theoretical coverage (1 - " + r'$\alpha$' + ")" + "(%)")
    plt.subplots_adjust(hspace=0.47)
    if save:
        if pdf_pgf == "pdf":
            plt.savefig(tds_dir + "/correlation_between_alpha_and_accuracy_and_set_size.pdf",
                        pad_inches=0, bbox_inches='tight')
        elif pdf_pgf == "pgf":
            plt.savefig(tds_dir + "/correlation_between_alpha_and_accuracy_and_set_size.pgf",
                        pad_inches=0, bbox_inches='tight')


def give_qualitative_result_allinone(path_init, tds_dir="../exp_data/eerst_paper_figures/",
                                     save=False, pdf_pgf="pdf", data_path="../data_group/"):
    fig = give_figure_specify_size(1.2, 0.5)
    ax_global = vis_utils.ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    dataset_names = ["Mineral (r)", "Mineral (p)", "Organic", "Bacteria"]
    for i in range(4):
        ax_g_0 = fig.add_subplot(2, 4, i + 1)
        ax_g_1 = fig.add_subplot(2, 4, i + 1 + 4)
        if i == 0:
            give_qualitative_result_rruff_raw(path_init, [ax_g_0, ax_g_1], data_path=data_path)
        elif i == 1:
            give_qualitative_result_rruff_preprocess(path_init, [ax_g_0, ax_g_1], data_path=data_path)
        elif i == 2:
            give_qualitative_result_organic(path_init, [ax_g_0, ax_g_1], data_path=data_path)
        elif i == 3:
            give_qualitative_result_bacteria(path_init, [ax_g_0, ax_g_1], data_path=data_path)
        if i == 0:
            ax_g_0.set_ylabel("Correct")
            ax_g_1.set_ylabel("Wrong")
        ax_g_0.set_title(dataset_names[i])
    ax_global.set_xlabel("\n Wavenumber (cm" + r"$^{-1})$")
    ax_global.set_ylabel("Intensity (a.u.)\n\n")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save:
        plt.savefig(tds_dir + "/qualitative_result.%s" % pdf_pgf, pad_inches=0, bbox_inches='tight')


def form3(x, pos):
    """ This function returns a string with 3 decimal places, given the input x"""
    return '%.1f' % x


def find_the_best_threshold_and_evaluate_accuracy(val_prediction, tt_ensemble,
                                                  selected_index,
                                                  reference_label_val,
                                                  val_label, reference_label_tt, tt_label, predicted_label_tt,
                                                  voting_number):
    """This function finds the best threshold (uncertainty) based on the validation dataset. Then we group
    the test predictions to low-uncertainty and high-uncertainty group and evaluate the matching accuracy under
    each group
    Args:
        val_prediction: [original_val, weighted_val]
        tt_ensemble: [original_tt_ensemble, weighted_tt_ensemble]
        selected_index: [selected index for original, selected index for the weighted]
        reference_label_val: the ground truth for the validation dataset
        val_label: the ground truth for the validation dataset
        reference_label_tt: the ground truth for the test dataset
        tt_label: the ground truth for the test data
        predicted_label_tt: the predicted label (it needs to be result after
        applying majority voting for the bacteria dataset)
        voting_number: the majority voting numbers
    """
    keys = list(val_prediction[0].keys())
    val_original_ensemble, \
    val_weighted_ensemble = np.zeros_like(val_prediction[0][keys[0]]), np.zeros_like(val_prediction[0][keys[0]])
    val_ensemble = [val_original_ensemble, val_weighted_ensemble]
    for i, s_stat in enumerate(val_prediction):
        for j, key in enumerate(s_stat.keys()):
            if j in selected_index[i]:
                val_ensemble[i] += s_stat[key]
    val_ensemble = [v / len(selected_index[0]) for v in val_ensemble]
    val_pred_baseon_class = [test.reorganize_similarity_score(v, reference_label_val)[0] for v in
                             val_ensemble]
    if len(voting_number) == 0:
        val_prediction = [reference_label_val[np.argmax(v, axis=-1)] for v in val_ensemble]
    else:
        val_prediction = []
        for i, s_val_pred in enumerate(val_ensemble):
            _, _pred_label = vis_utils.majority_voting(s_val_pred, reference_label_val,
                                                       val_label, voting_number[i])
            val_prediction.append(_pred_label)
    val_threshold = []
    for i in range(2):
        correct_or_wrong = np.array([0 if v == q else 1 for v, q in zip(val_prediction[i], val_label)])
        if i == 0:
            norm_pred = softmax(val_pred_baseon_class[i], axis=-1)
        else:
            norm_pred = val_pred_baseon_class[i]
        selected_predict = norm_pred[np.arange(len(val_label)), val_prediction[i]]
        _nll = -np.log(selected_predict)
        fpr, tpr, thresholds = roc_curve(correct_or_wrong, _nll)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        val_threshold.append(optimal_threshold)
    stat_baseon_uncertainty = np.zeros([2, 4])
    for i in range(2):
        tt_pred_baseon_class, _ = test.reorganize_similarity_score(tt_ensemble[i],
                                                                   reference_label_tt)
        if i == 0:
            tt_pred_baseon_class = softmax(tt_pred_baseon_class, axis=-1)
        select_predict = tt_pred_baseon_class[np.arange(len(tt_label)), predicted_label_tt[i]]
        _nll = -np.log(select_predict)
        correct_or_wrong = np.array([0 if v == q else 1 for v, q in zip(predicted_label_tt[i], tt_label)])

        high_uncertainty_index = np.where(_nll >= val_threshold[i])[0]
        high_uncertainty_correct = len(high_uncertainty_index) - np.sum(correct_or_wrong[high_uncertainty_index])

        low_uncertainty_index = np.where(_nll < val_threshold[i])[0]
        low_uncertainty_correct = len(low_uncertainty_index) - np.sum(correct_or_wrong[low_uncertainty_index])
        stat_baseon_uncertainty[i, :] = [low_uncertainty_correct, len(low_uncertainty_index),
                                         high_uncertainty_correct, len(high_uncertainty_index)]
    return stat_baseon_uncertainty, val_threshold


def _give_uncertainty_distribution_for_single_dataset(dataset, raman_type,
                                                      num_select, voting_number, uncertainty, prediction_status,
                                                      split_version=100, qualitative_study=False, path_init="../",
                                                      get_similarity=False, data_path="../data_group/", strategy="sigmoid"):
    path2load = path_init + "/%s/" % raman_type + "/tds/"
    folder2read = [v for v in os.listdir(path2load) if os.path.isdir(path2load + v) and "split_%d" % split_version in v]
    dir2load_data = path_init + "/%s/" % raman_type
    dir2load_data = [dir2load_data + "/" + v + "/data_splitting/" for v in os.listdir(dir2load_data) if
                     "tds" not in v and "version_%d" % split_version in v][0]
    folder2read = folder2read[0]
    original_weight_stat = ["original", "weighted"]
    folder2read = path2load + folder2read
    val_prediction = pickle.load(open(folder2read + "/validation_prediction.obj", "rb"))
    tt_prediction = pickle.load(open(folder2read + "/test_prediction.obj", "rb"))
    original_val, weighted_val = val_prediction
    original_tt, weighted_tt = tt_prediction
    
    args = const.give_args_test(raman_type=raman_type)
    args["pre_define_tt_filenames"] = True
    validation_accuracy = np.zeros([len(list(original_val.keys())) - 1, 2])
    if dataset == "RRUFF" or dataset == "ORGANIC":
        if dataset == "RRUFF":
            tr_data, tt_data, _, label_name_tr = test.get_data(args, dir2load_data, read_twin_triple="cls",
                                                               print_info=False, dir2read=data_path)
        else:
            tr_data, tt_data, _, label_name_tr = test.get_data(args, dir2load_data, read_twin_triple="cls",
                                                               print_info=False, dir2read=data_path)
        fake_val_reference, fake_val_data = pdd.get_fake_reference_and_test_data(tr_data, 1, data=dataset)
        reference_val_label, val_label = fake_val_reference[1], fake_val_data[1]
        for j, key in enumerate(list(original_val.keys())[:-1]):
            _val_pred = original_val[key]
            if strategy == "sigmoid" or strategy == "sigmoid_softmax":
                _val_pred = expit(_val_pred)        
            _correct = np.sum(fake_val_reference[1][np.argmax(_val_pred, axis=-1)] == fake_val_data[1]) / len(
                fake_val_data[0])
            validation_accuracy[j, 0] = _correct
        for j, key in enumerate(list(weighted_val.keys())[:-1]):
            _val_pred = weighted_val[key]
            _correct = np.sum(fake_val_reference[1][np.argmax(_val_pred, axis=-1)] == fake_val_data[1]) / len(
                fake_val_data[0])
            validation_accuracy[j, 1] = _correct
    else:
        tr_data, val_data, tt_data, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls",
                                                                     print_info=False, dir2read=data_path)
        reference_val_label, val_label = tr_data[1], val_data[1]
        for m, stat in enumerate([original_val, weighted_val]):
            for j, key in enumerate(list(stat.keys())[:-1]):
                if m == 0:
                    if strategy == "sigmoid" or strategy == "sigmoid_softmax":
                        _val_pred = expit(stat[key])
                    else:                    
                        _val_pred = stat[key]
                _correct = np.sum(tr_data[1][np.argmax(_val_pred, axis=-1)] == val_data[1]) / len(val_data[1])
                validation_accuracy[j, m] = _correct
    original_select = np.argsort(validation_accuracy[:, 0])[-num_select:]
    weighted_select = np.argsort(validation_accuracy[:, 1])[-num_select:]
    for j, key in enumerate(list(original_tt.keys())):
        if j == 0:
            original_tt_ensemble = np.zeros_like(original_tt[key])
        if strategy == "sigmoid" or strategy == "sigmoid_softmax":
            original_tt[key] = expit(original_tt[key])    
        if j in original_select:
            original_tt_ensemble += original_tt[key]
    original_tt_ensemble /= len(original_select)
    for j, key in enumerate(list(weighted_tt.keys())):
        if j == 0:
            weighted_tt_ensemble = np.zeros_like(weighted_tt[key])
        if j in weighted_select:
            weighted_tt_ensemble += weighted_tt[key]
    weighted_tt_ensemble /= len(weighted_select)
    predicted_label_on_test_data = []
    correspond_tr_index = []
    for j, single_stat in enumerate([original_tt_ensemble, weighted_tt_ensemble]):
        if dataset != "BACTERIA":
            _pred_label = tr_data[1][np.argmax(single_stat, axis=-1)]
            accuracy = np.sum(_pred_label == np.array(tt_data[1])) / len(tt_data[0])
        else:
            accuracy, _pred_label = vis_utils.majority_voting(single_stat, tr_data[1],
                                                              tt_data[1], voting_number[j])
        pred_baseon_class, corr_tr_index = test.reorganize_similarity_score(single_stat,
                                                                            tr_data[1])
        if strategy == "softmax":
            pred_baseon_class = softmax(pred_baseon_class, axis=-1)
        _nll_prediction = pred_baseon_class[np.arange(len(tt_data[0])), _pred_label]
        print("NLL prediction", np.max(_nll_prediction), np.min(_nll_prediction))
        _nll_score = _nll_prediction
        if split_version == 100:
            uncertainty.update({"%s_%s_%s" % (dataset, raman_type, original_weight_stat[j]): _nll_score})
        else:
            uncertainty.update({"%s_%s_%s_version_%d" % (dataset, raman_type, original_weight_stat[j],
                                                         split_version): _nll_score})

        _pred_stat = np.concatenate([np.expand_dims(tt_data[1], axis=-1),
                                     np.expand_dims(_pred_label, axis=-1)], axis=-1)
        if split_version == 100:
            prediction_status.update({"%s_%s_%s" % (dataset, raman_type, original_weight_stat[j]): _pred_stat})
        else:
            prediction_status.update({"%s_%s_%s_version_%d" % (dataset, raman_type, original_weight_stat[j],
                                                               split_version): _pred_stat})
        print("%s + %s + %s : %.4f" % (dataset, raman_type, original_weight_stat[j], accuracy))
        predicted_label_on_test_data.append(_pred_label)
        correspond_tr_index.append(corr_tr_index)

    accuracy_baseon_uncertainty, \
    optimal_threshold = find_the_best_threshold_and_evaluate_accuracy([original_val, weighted_val],
                                                                      [original_tt_ensemble, weighted_tt_ensemble],
                                                                      [original_select, weighted_select],
                                                                      reference_val_label,
                                                                      val_label,
                                                                      tr_data[1], tt_data[1],
                                                                      predicted_label_on_test_data, voting_number)
    if not qualitative_study:
        return uncertainty, prediction_status, accuracy_baseon_uncertainty, optimal_threshold
    else:
        if not get_similarity:
            return uncertainty, prediction_status, correspond_tr_index, \
                   optimal_threshold, tr_data, tt_data, label_name_tr, np.arange(args["max_wave"])[args["min_wave"]:]
        else:
            return original_val, original_tt_ensemble, original_select, \
                   reference_val_label, val_label, tr_data[1], tt_data[1]


def give_original_weight_uncertainty(uncertainty, prediction_status, dataset, use_nll_or_prob="nll"):
    stat_orig, stat_weight = {}, {}
    min_value = 0
    high_value = [6 if use_nll_or_prob == "nll" else 1][0]
    if dataset == "RRUFF_R":
        num_bins = 5  # 8
    elif dataset == "RRUFF_P":
        num_bins = 5
    elif dataset == "ORGANIC":
        num_bins=3
    else:
        num_bins = 7
    uncertainty_array, prediction_array = [], []
    for key in uncertainty.keys():
        predict_prob = uncertainty[key]
        print(key, np.max(predict_prob), np.min(predict_prob))
        _stat = group_uncertainty_and_prediction(predict_prob,
                                                 prediction_status[key],
                                                 min_value, high_value, num_bins, False)
        if "weight" in key:
            stat_weight[key] = _stat
        else:
            stat_orig[key] = _stat
            prediction_array.append(prediction_status[key])
            uncertainty_array.append(predict_prob)
    if dataset == "RRUFF_Rs" or dataset == "RRUFF_Ps" or dataset == "ORGANICs":
        return stat_weight
    else:
        return stat_orig, prediction_array, uncertainty_array


def give_avg_std_for_uncertainty(stat_weight):
    stat = [[] for _ in range(3)]
    max_dim = np.max([np.shape(stat_weight[key])[1] for key in stat_weight.keys()])
    for key in stat_weight.keys():
        _value = stat_weight[key]
        if np.shape(_value)[1] < max_dim:
            _value = np.concatenate([_value, np.zeros([len(_value), max_dim - np.shape(_value)[1]])],
                                    axis=-1)
        for j in range(3):
            stat[j].append(_value[j])
    for j, v in enumerate(stat):
        stat[j] = np.array(v)
    tot = stat[1] + stat[2]
    tot[tot == 0] = 1
    stat_c_percent = stat[1] / tot
    stat_w_percent = stat[2] / tot
    percent_stat = [stat_c_percent, stat_w_percent]
    stat_avg, stat_std = [], []
    for j in range(3):
        if j == 0:
            x_avg = np.sum(stat[0], axis=0) / np.sum(stat[0] != 0, axis=0)
        else:
            _divide = np.sum(percent_stat[j - 1] != 0, axis=0)
            _divide[_divide == 0] = 1
            x_avg = np.sum(percent_stat[j - 1], axis=0) / _divide
        stat_avg.append(x_avg)
        x_std = np.zeros_like(x_avg)
        for m in range(np.shape(stat[0])[1]):
            if j == 0:
                v = stat[j][:, m]
            else:
                v = percent_stat[j - 1][:, m]
            if len(v[v != 0]) > 0:
                if np.sum(v[v != 0]) != 0:
                    x_std[m] = 1.95 * np.std(v[v != 0]) / np.sqrt(np.sum(v != 0))
        stat_std.append(x_std)
    return stat_avg, stat_std


def give_calibration_single_score_prediction(prediction, apply_softmax, label):
    if apply_softmax == "softmax":
        prediction = softmax(prediction, axis=-1)
    elif apply_softmax == "sigmoid":
        prediction = expit(prediction)
    prediction_score = prediction[np.arange(len(prediction)), label]
    return prediction_score


def give_test_prediction_baseon_single_score_threshold(prediction, apply_softmax, label, threshold, show=False):
    if apply_softmax == "softmax":
        prediction = softmax(prediction, axis=-1)
    elif apply_softmax == "sigmoid":
        prediction = expit(prediction)
    prediction_select = [np.where(v >= threshold)[0] for v in prediction]
    prediction_select = [v if len(v) > 0 else [np.argmax(prediction[i])] for i, v in enumerate(prediction_select)]
    accuracy = [1 for i, v in enumerate(prediction_select) if label[i] in v]
    if show:
        print("Matching accuracy %.2f" % (np.sum(accuracy) / len(label)))
    return prediction_select, np.sum(accuracy) / len(label)


def main_plot_for_scoring_rule(path2load, data_split, raman_type, dataset, use_original_or_weighted,
                               alpha_group, show=False, data_path="../data_group/", apply_softmax=True):
    statistics_group = np.zeros([len(data_split), len(alpha_group), 2])
    if dataset == "RRUFF":
        args = const.give_args_test(raman_type=raman_type)
        args["pre_define_tt_filenames"] = False
        tr_data, tt_data, _, label_name_tr = test.get_data(args, None, read_twin_triple="cls",
                                                           print_info=False, dir2read=data_path)
        [_, reference_val_label], [_, val_label] = pdd.get_fake_reference_and_test_data(tr_data, 1, data=dataset)
        tr_label_group = [reference_val_label, tr_data[1]]
        tt_label = tt_data[1]
    elif dataset == "BACTERIA":
        args = const.give_args_test(raman_type="bacteria_random_reference_finetune")
        args["pre_define_tt_filenames"] = False
        tr_data, val_data, tt_data, _, _ = test.get_data(args, None, read_twin_triple="cls", print_info=False,
                                                         dir2read=data_path)
        tr_label_group = [tr_data[1], tr_data[1]]
        val_label, tt_label = val_data[1], tt_data[1]
    else:
        args = const.give_args_test(raman_type=raman_type)
        args["pre_define_tt_filenames"] = True
    for split_index, s_split in enumerate(data_split):
        path = path2load + [v for v in os.listdir(path2load) if "split_%d" % s_split in v and ".txt" not in v][0] + "/"
        # path = path2load + "split_%d/" % s_split
        val_prediction = pickle.load(open(path + "validation_prediction.obj", "rb"))
        tt_prediction = pickle.load(open(path + "test_prediction.obj", "rb"))
        if use_original_or_weighted == "original":
            val_pred_en, tt_pred_en = val_prediction[0]["ensemble_avg"], tt_prediction[0]["ensemble_avg"]
        elif use_original_or_weighted == "weighted":
            val_pred_en, tt_pred_en = val_prediction[1]["ensemble_avg"], tt_prediction[1]["ensemble_avg"]
        if dataset == "ORGANIC":
            _fake_reference_label, val_label, _tr_label, tt_label = _get_dir_update(path2load, s_split, args,
                                                                                    data_path=data_path)
            tr_label_group = [_fake_reference_label, _tr_label]
        val_pred_baseon_cls, _ = test.reorganize_similarity_score(val_pred_en, tr_label_group[0])
        tt_pred_baseon_cls, _ = test.reorganize_similarity_score(tt_pred_en, tr_label_group[1])
        # Theoretical accuracy
        _tt_top1 = np.argmax(tt_pred_baseon_cls, axis=1)
        _compare = np.sum([v == q for v, q in zip(_tt_top1, tt_label)])
        # print("Theoretical accuracy", _compare / len(_tt_top1))
        print(np.max(val_pred_baseon_cls), np.min(val_pred_baseon_cls))
        _statistics = give_plot_for_first_scoring_rule(val_pred_baseon_cls,
                                                       tt_pred_baseon_cls,
                                                       val_label, tt_label,
                                                       alpha_group=alpha_group, ax=None, show=show, apply_softmax=apply_softmax)
        statistics_group[split_index] = _statistics

    if show:
        statistics_avg = np.mean(statistics_group, axis=0)
        statistics_conf = np.std(statistics_group, axis=0)  # / np.sqrt(len(data_split))
        vis_utils.show_twinx(alpha_group, statistics_avg[:, 0], statistics_avg[:, 1], None)
    return statistics_group


def get_path_for_conformal(path_init, raman_type):
    path2load = [path_init + v for v in os.listdir(path_init) if v == raman_type][0] + "/tds/"
    split_version = [int(v.split("split_")[1].split(".txt")[0]) for v in os.listdir(path2load) if
                     "split_" in v and '.txt' in v]
    return path2load, split_version


def give_plot_for_first_scoring_rule(val_prediction, test_prediction, val_label, tt_label,
                                     alpha_group, ax=None, show=False, apply_softmax=True):
    val_prediction_score = give_calibration_single_score_prediction(val_prediction, apply_softmax, val_label)
    stat_group = np.zeros([len(alpha_group), 2])
    for i, s_alpha in enumerate(alpha_group):
        threshold = np.quantile(val_prediction_score, s_alpha)
        tt_prediction, tt_accuracy = give_test_prediction_baseon_single_score_threshold(test_prediction,
                                                                                        apply_softmax, tt_label,
                                                                                        threshold)
        num_selection = np.mean([len(v) for v in tt_prediction])
        stat_group[i, :] = [tt_accuracy, num_selection]
    if show:
        vis_utils.show_twinx(alpha_group, stat_group[:, 0], stat_group[:, 1], None)
    return stat_group


def _get_dir_update(path_init, split_version, args, data_path="../data_group/"):
    path2load = path_init.split("/tds")[0] + "/"
    dir2load = path2load + [v for v in os.listdir(path2load) if "version_%d_" % split_version in v][0] \
               + "/data_splitting/"
    tr_data, tt_data, _, _ = test.get_data(args, dir2load, read_twin_triple="cls", print_info=False,
                                           dir2read=data_path)
    fake_val_reference, fake_val_data = pdd.get_fake_reference_and_test_data(tr_data, 1, data="ORGANIC")
    reference_val_label, val_label = fake_val_reference[1], fake_val_data[1]
    return reference_val_label, val_label, tr_data[1], tt_data[1]


def get_multiple_rruff_uncertainty(raman_type="raw", path_init="../", 
                                   use_nll_or_prob="nll", data_path="../data_group/",
                                   strategy="sigmoid"):
    dataset = "RRUFF"
    if raman_type == "raw":
        num_select = 5
        version_use = [2, 3, 5, 6]
    elif raman_type == "excellent_unoriented":
        num_select = 4
        version_use = [0, 1, 3, 4]
    uncertainty, prediction_status = {}, {}
    accu_on_original, accu_on_weighed = np.zeros([len(version_use), 4]), np.zeros([len(version_use), 4])
    for index, i in enumerate(version_use):
        uncertainty, prediction_status, \
        accu_on_uncert, threshold = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                                      raman_type,
                                                                                      num_select,
                                                                                      [], uncertainty,
                                                                                      prediction_status,
                                                                                      split_version=i,
                                                                                      path_init=path_init,
                                                                                      data_path=data_path,
                                                                                      strategy=strategy)
        accu_on_uncert[:, 0] = accu_on_uncert[:, 0] / accu_on_uncert[:, 1] * 100
        accu_on_uncert[:, 2] = accu_on_uncert[:, 2] / accu_on_uncert[:, 3] * 100
        accu_on_original[index, :] = accu_on_uncert[0, :]
        accu_on_weighed[index, :] = accu_on_uncert[1, :]
    name = ["RRUFF_R" if raman_type == "raw" else "RRUFF_P"][0]
    stat_weight, pred_array, uncert_array = give_original_weight_uncertainty(uncertainty, prediction_status, dataset=name,
                                                   use_nll_or_prob=use_nll_or_prob)
    stat_avg, stat_std = give_avg_std_for_uncertainty(stat_weight)
    # label_group = ["original", "weighted"]
    # for i, s_accu in enumerate([accu_on_original, accu_on_weighed]):
    #     print("-----------------------%s----------------------" % label_group[i])
    #     avg_accu = np.mean(s_accu, axis=0)
    #     std_accu = 1.95 * np.std(s_accu, axis=0) / np.sqrt(len(s_accu))
    #     print("Within %.1f ± %.1f low uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[1], std_accu[1],
    #                                                                                      avg_accu[0], std_accu[0]))
    #     print("Within %.1f ± %.1f high uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[3], std_accu[3],
    #                                                                                       avg_accu[2], std_accu[2]))

    return stat_weight, stat_avg, stat_std, [pred_array, uncert_array]



def get_multiple_organic_uncertainty(organic_type, data_path, path_init="/scratch/blia/", use_nll_or_prob="nll",
                                     strategy="sigmoid"):
    dataset = "ORGANIC"
    raman_type = organic_type
    siamese_version = 60
    pos_neg_ratio = 1
    num_select = 5
    if organic_type == "organic_target_raw":
        version_use = [3, 4, 5]  # 3
    elif organic_type == "organic_target":
        version_use = [0, 2, 4, 5]  # 4
    uncertainty, prediction_status = {}, {}
    accu_on_original, accu_on_weighed = np.zeros([len(version_use), 4]), np.zeros([len(version_use), 4])
    for index, i in enumerate(version_use):
        uncertainty, prediction_status, accu_on_uncert, threshold = _give_uncertainty_distribution_for_single_dataset(
            dataset,
            # siamese_version,
            raman_type,
            # pos_neg_ratio,
            num_select,
            [], uncertainty,
            prediction_status,
            split_version=i, path_init=path_init, data_path=data_path,
                                                                                      strategy=strategy)
        for j in range(len(accu_on_uncert)):
            if accu_on_uncert[j, 1] != 0:
                accu_on_uncert[j, 0] = accu_on_uncert[j, 0] / accu_on_uncert[j, 1] * 100
            if accu_on_uncert[j, 3] != 0:
                accu_on_uncert[j, 2] = accu_on_uncert[j, 2] / accu_on_uncert[j, 3] * 100
            else:
                accu_on_uncert[j, 2] = 100
        accu_on_original[index, :] = accu_on_uncert[0, :]
        accu_on_weighed[index, :] = accu_on_uncert[1, :]
    stat_weight, pred_array, uncert_array = give_original_weight_uncertainty(uncertainty, prediction_status, dataset="ORGANIC",
                                                   use_nll_or_prob=use_nll_or_prob)
    stat_avg, stat_std = give_avg_std_for_uncertainty(stat_weight)
    label_group = ["original", "weighted"]
    for i, s_accu in enumerate([accu_on_original, accu_on_weighed]):
        print("-----------------------%s----------------------" % label_group[i])
        print(s_accu)
        avg_accu = np.mean(s_accu, axis=0)
        std_accu = 1.95 * np.std(s_accu, axis=0) / np.sqrt(len(s_accu))
        print("Within %d ± %d low uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[1], std_accu[1],
                                                                                     avg_accu[0], std_accu[0]))
        print("Within %d ± %d high uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[3], std_accu[3],
                                                                                      avg_accu[2], std_accu[2]))
    return stat_weight, stat_avg, stat_std, [pred_array, uncert_array]



def get_multiple_bacteria_uncertainty(path_init="/scratch/blia/", use_nll_or_prob="nll", data_path="../data_group/",
                                      strategy="sigmoid"):
    dataset = "BACTERIA"
    raman_type = ["bacteria_reference_finetune" for _ in range(4)]
    version_use = [0, 1, 3, 4]
    num_select = [8, 5, 5, 5]
    voting_number = [[37, 14],
                     [55, 21],
                     [32, 21],
                     [15, 25]]
    uncertainty, prediction_status = {}, {}
    accu_on_original, accu_on_weighed = np.zeros([len(version_use), 4]), np.zeros([len(version_use), 4])
    for i in range(4):
        uncertainty, prediction_status, \
        accu_on_uncert, threshold = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                                      raman_type[i],
                                                                                      num_select[i],
                                                                                      voting_number[i],
                                                                                      uncertainty,
                                                                                      prediction_status,
                                                                                      split_version=version_use[i],
                                                                                      path_init=path_init,
                                                                                      data_path=data_path,
                                                                                      strategy=strategy)
        accu_on_uncert[:, 0] = accu_on_uncert[:, 0] / accu_on_uncert[:, 1] * 100
        accu_on_uncert[:, 2] = accu_on_uncert[:, 2] / accu_on_uncert[:, 3] * 100
        accu_on_original[i, :] = accu_on_uncert[0, :]
        accu_on_weighed[i, :] = accu_on_uncert[1, :]
    stat_weight, pred_array, uncert_array = give_original_weight_uncertainty(uncertainty, prediction_status, dataset="Bacteria",
                                                   use_nll_or_prob=use_nll_or_prob)
    stat_avg, stat_std = give_avg_std_for_uncertainty(stat_weight)
    
    # label_group = ["original", "weighted"]
    # for i, s_accu in enumerate([accu_on_original, accu_on_weighed]):
    #     print("-----------------------%s----------------------" % label_group[i])
    #     avg_accu = np.mean(s_accu, axis=0)
    #     std_accu = 1.95 * np.std(s_accu, axis=0) / np.sqrt(len(s_accu))
    #     print("Within %.1f ± %d low uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[1], std_accu[1],
    #                                                                                    avg_accu[0], std_accu[0]))
    #     print("Within %.1f ± %d high uncertain predictions, %.2f ± %.2f are correct" % (avg_accu[3], std_accu[3],
    #                                                                                     avg_accu[2], std_accu[2]))

    return stat_weight, stat_avg, stat_std, [pred_array, uncert_array]


def plot_fillx_filly(x_avg, x_std, y_avg, y_std, ax, color_use):
    index = np.where(y_avg != 0)[0]
    x_avg, x_std = x_avg[index], x_std[index]
    y_avg, y_std = y_avg[index], y_std[index]
    ax.plot(x_avg, y_avg, color_use)


def group_uncertainty_and_prediction(uncertainty, prediction, min_value, max_value, num_bins=100, show=True):
    """This function groups the uncertainty into bins and then check the correct/wrong predictions within each bin
    Args:
        uncertainty: [number of test data]
        prediction: [number of test data, 2]
    """
    pred_label = (np.diff(prediction, axis=-1).squeeze(1) == 0).astype('int32')
    group = np.linspace(np.min(uncertainty), np.max(uncertainty), num_bins + 1)
    # group = np.linspace(min_value, max_value, num_bins + 1)
    uncertainty_avg, correct_avg, wrong_avg = [], [], []
    for i in range(num_bins):
        low = group[i]
        high = group[i + 1]
        index = np.where(np.logical_and(uncertainty >= low, uncertainty < high))[0]
        if len(index) > 0:
            uncertainty_avg.append(np.mean(uncertainty[index]))
            correct_avg.append(np.sum(pred_label[index]))
            wrong_avg.append(len(index) - np.sum(pred_label[index]))
    stat = [uncertainty_avg, correct_avg, wrong_avg]
    
    for i, s in enumerate(stat):
        stat[i] = np.array(s)
    if show:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        ax.plot(uncertainty_avg, stat[1] / len(pred_label))
    return stat


def show_accu(accu, use_name, return_value=False, print_info=False):
    label_group = ["original", "weighted"]
    accu_avg = np.mean(accu, axis=-1)
    print("The shape of the accuracy", np.shape(accu))
    
    accu_conf = 1.95 * np.std(accu, axis=-1) / np.sqrt(np.shape(accu)[1])
    if print_info:
        for i_index, j in enumerate([0, 2]):
            print(use_name, label_group[i_index], ": %.2f +- %.2f" % (accu_avg[j], accu_conf[j]))
        print("-----------------------------------------------------------")
    if return_value:
        return accu_avg, accu_conf


def _give_rruff_raw_single_experiment(path_init="../exp_data/exp_group/", data_path="../data_group/"):
    dataset = "RRUFF"
    raman_type = "raw"
    num_select = 5
    _output = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                raman_type,
                                                                num_select, [], {}, {},
                                                                split_version=2,
                                                                qualitative_study=True, path_init=path_init,
                                                                data_path=data_path)
    keys = list(_output[0].keys())
    return [_output[0][key] for key in keys], [_output[1][key] for key in keys], _output[2:]


def _give_rruff_preprocess_single_experiment(path_init="../", data_path="../data_group/"):
    dataset = "RRUFF"
    raman_type = "excellent_unoriented"
    num_select = 5
    _output = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                raman_type,
                                                                num_select, [], {}, {},
                                                                split_version=0, qualitative_study=True,
                                                                path_init=path_init,
                                                                data_path=data_path)
    keys = list(_output[0].keys())
    return [_output[0][key] for key in keys], [_output[1][key] for key in keys], _output[2:]


def _give_organic_single_experiment(select_index=0, path_init="../", data_path="../data_group/"):
    """Args:
        select_index: 0 organic_target_raw, 1: organic_target
    """
    dataset = "ORGANIC"
    raman_type = ["organic_target_raw", "organic_target"]
    num_select = [0, 0]
    split_version = 3
    _output = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                raman_type[select_index],
                                                                num_select[select_index], [], {}, {},
                                                                split_version=split_version,
                                                                qualitative_study=True,
                                                                path_init=path_init,
                                                                data_path=data_path)
    keys = list(_output[0].keys())
    return [_output[0][key] for key in keys], [_output[1][key] for key in keys], _output[2:]


def _give_bacteria_single_experiment(path_init="../", data_path="../data_group/"):
    dataset = "BACTERIA"
    raman_type = "bacteria_reference_finetune"
    num_select = 8
    voting_number = [37, 14]
    _output = _give_uncertainty_distribution_for_single_dataset(dataset,
                                                                raman_type,
                                                                num_select, voting_number, {}, {},
                                                                split_version=0,
                                                                qualitative_study=True, path_init=path_init,
                                                                data_path=data_path)
    keys = list(_output[0].keys())
    return [_output[0][key] for key in keys], [_output[1][key] for key in keys], _output[2:]


def give_correct_wrong_index(uncertainty, optimal_threshold, prediction, use_version):
    correct_wrong = (np.diff(prediction[use_version], axis=-1)[:, 0] == 0).astype(np.int32)
    low_uncert_index = np.where(uncertainty[use_version] < optimal_threshold[use_version])[0]
    high_uncert_index = np.where(uncertainty[use_version] > optimal_threshold[use_version])[0]
    low_uncert_correct = low_uncert_index[correct_wrong[low_uncert_index] == 1]
    low_uncert_wrong = low_uncert_index[correct_wrong[low_uncert_index] == 0]
    high_uncert_correct = high_uncert_index[correct_wrong[high_uncert_index] == 1]
    high_uncert_wrong = high_uncert_index[correct_wrong[high_uncert_index] == 0]
    print("Within low uncertain predictions: %d are correct and %d are wrong" % (len(low_uncert_correct),
                                                                                 len(low_uncert_wrong)))
    print("Within high uncertain predictions: %d are correct and %d are wrong" % (len(high_uncert_correct),
                                                                                  len(high_uncert_wrong)))
    return [low_uncert_correct, low_uncert_wrong], [high_uncert_correct, high_uncert_wrong]


def find_loc_to_write(spectrum, wavenumber, sliding_window=400, specify_loc=700):
    num_seg = np.linspace(0, len(spectrum), len(spectrum) // sliding_window).astype('int32')
    spectrum = abs(np.diff(spectrum))
    sum_value = [np.sum(spectrum[num_seg[i]:num_seg[i + 1]]) for i in range(len(num_seg) - 1)]
    small = num_seg[np.argmin(sum_value)]
    if len(wavenumber) == 1000 or specify_loc != 0:
        return specify_loc
    else:
        return small + wavenumber[0]


def give_single_plot(tr_data, tt_data, prediction, correspond_tr_index,
                     label_name_tr, wavenumber, index_use, use_version, ax=None):
    if not ax:
        fig = plt.figure(figsize=(5, 2.5))
        ax = fig.add_subplot(111)
    scale = 1.4
    if len(label_name_tr) == 30:
        x_loc = 450
    else:
        x_loc = 50
    if "class" in label_name_tr[0]:
        label_name_tr = [v.replace("class", "organic") for v in label_name_tr]
    tt_input_label = tt_data[1][index_use]
    predict_label = prediction[use_version][index_use, 1]
    _pred_tr_index = correspond_tr_index[use_version][index_use, predict_label]
    _corr_tr_index = np.where(tr_data[1] == tt_input_label)[0]
    legend_group = ["Ref: %s" % label_name_tr[tt_input_label], "Input: %s" % label_name_tr[tt_input_label],
                    "Pred: %s" % label_name_tr[predict_label]]
    if len(_corr_tr_index) > 45:
        _avg_value = np.mean(tr_data[0][_corr_tr_index], axis=0)
        _conf = np.std(tr_data[0][_corr_tr_index], axis=0)
        ax.plot(wavenumber, _avg_value, 'r', lw=0.5)
        ax.fill_between(wavenumber, (_avg_value - _conf), (_avg_value + _conf), color='r', alpha=0.5)
    else:
        if len(_corr_tr_index) > 5:
            _corr_tr_index = np.random.choice(_corr_tr_index, 5, replace=False)
        for s in _corr_tr_index:
            ax.plot(wavenumber, tr_data[0][s], 'r', alpha=0.9, lw=0.5)
    ax.text(x_loc, scale - 0.3, legend_group[0], color='r')
    ax.plot(wavenumber, tt_data[0][index_use] + scale, 'g', alpha=0.8, lw=0.5)
    ax.text(x_loc, 2 * scale - 0.1, legend_group[1], color='g')
    ax.plot(wavenumber, tr_data[0][_pred_tr_index] + scale, 'b', alpha=0.8, lw=0.5)
    ax.text(x_loc, 2 * scale - 0.4, legend_group[2], color='b')
    ax.set_ylim((-0.1, 2 * scale + 0.2))
    ax.yaxis.set_major_formatter(plt.NullFormatter())


def give_single_column(tr_data, tt_data, prediction, correspond_tr_index, label_name,
                       wavenumber, indices, use_version, tds_dir=None,
                       save=False, save_name=None, ax_g=[]):
    if len(ax_g) == 0:
        fig = plt.figure(figsize=(4, 7 / 3 * 2.5))
        ax_g = [fig.add_subplot(3, 1, i + 1) for i in range(len(indices))]
        plt.subplots_adjust(hspace=0.05)
    for i, s_index in enumerate(indices):
        ax = ax_g[i]
        give_single_plot(tr_data, tt_data, prediction, correspond_tr_index,
                         label_name, wavenumber, s_index,
                         use_version, ax=ax)
        if i != len(indices) - 1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
    if save:
        dpi = [500 if "bacteria" not in save_name else 400][0]
        plt.savefig(tds_dir + "/visualize_%s.jpg" % save_name, pad_inches=0, bbox_inches='tight',
                    dpi=dpi)


def give_qualitative_result_rruff_raw(path_init, ax_g=[], tds_dir=None, save=False, data_path="../data_group/"):
    uncertainty, prediction, statistics = _give_rruff_raw_single_experiment(path_init, data_path=data_path)
    correspond_tr_index, optimal_threshold, tr_data, tt_data, label_name_tr, wavenumber = statistics
    low_uncert_stat, high_uncert_stat = give_correct_wrong_index(uncertainty, optimal_threshold, prediction, 1)
    low_uncert_correct, low_uncert_wrong = low_uncert_stat
    give_single_column(tr_data, tt_data, prediction, correspond_tr_index, label_name_tr,
                       wavenumber, [low_uncert_correct[3], low_uncert_wrong[1]],
                       1,
                       tds_dir=tds_dir, save=save,
                       save_name="RRUFF_raw_low_uncertainty", ax_g=ax_g)


def give_qualitative_result_rruff_preprocess(path_init, ax_g=[], tds_dir=None, save=False,
                                             data_path="../data_group/"):
    uncertainty, prediction, statistics = _give_rruff_preprocess_single_experiment(path_init, data_path=data_path)
    correspond_tr_index, optimal_threshold, tr_data, tt_data, label_name_tr, wavenumber = statistics
    low_uncert_stat, high_uncert_stat = give_correct_wrong_index(uncertainty, optimal_threshold, prediction, 1)
    high_uncert_correct, high_uncert_wrong = high_uncert_stat
    give_single_column(tr_data, tt_data, prediction, correspond_tr_index, label_name_tr,
                       wavenumber, [high_uncert_correct[2], high_uncert_wrong[0]],
                       1,
                       tds_dir=tds_dir, save=save,
                       save_name="RRUFF_preprocess_high_uncertainty", ax_g=ax_g)


def give_qualitative_result_organic(path_init, ax_g=[], tds_dir=None, save=False, data_path="../data_group/"):
    uncertainty, prediction, statistics = _give_organic_single_experiment(0, path_init,
                                                                          data_path=data_path)
    correspond_tr_index, optimal_threshold, tr_data, tt_data, label_name_tr, wavenumber = statistics
    low_uncert_stat, high_uncert_stat = give_correct_wrong_index(uncertainty, optimal_threshold, prediction, 1)
    low_uncert_correct, low_uncert_wrong = low_uncert_stat
    use_version = 1
    wavenumber = np.linspace(106.62457839661, 3416.04065695651, np.shape(tr_data[0])[1])
    give_single_column(tr_data, tt_data, prediction, correspond_tr_index, label_name_tr,
                       wavenumber, [low_uncert_correct[2],
                                    low_uncert_wrong[0]],  # low_uncert_wrong[0],
                       use_version, tds_dir=tds_dir,
                       save=save, save_name="organic_low_uncertainty",
                       ax_g=ax_g)


def give_qualitative_result_bacteria(path_init, ax_g=[], tds_dir=None, save=False, data_path="../data_group/"):
    uncertainty, prediction, statistics = _give_bacteria_single_experiment(path_init,
                                                                           data_path=data_path)
    correspond_tr_index, optimal_threshold, tr_data, tt_data, label_name_tr, wavenumber = statistics
    wavenumber = np.load("../bacteria/wavenumbers.npy")
    low_uncert_stat, high_uncert_stat = give_correct_wrong_index(uncertainty, optimal_threshold, prediction, 1)
    high_uncert_correct, high_uncert_wrong = high_uncert_stat
    use_version = 1
    give_single_column(tr_data, tt_data, prediction, correspond_tr_index, label_name_tr,
                       wavenumber, [high_uncert_correct[40],
                                    high_uncert_wrong[53]],  # high_uncert_wrong[53], #271
                       use_version,
                       tds_dir=tds_dir,
                       save=save, save_name="bacteria_high_uncertainty",
                       ax_g=ax_g)


# --- Table 2
def report_accuracy_table(path_init="../"):
    print("-----------------RRUFF Raw accuracy-----------------------")
    accu = vis_utils.compare_accuracy_simple("raw", 1, path_init)
    show_accu(accu, "rruff (r)", print_info=True)

    print("-----------------RRUFF preprocessed accuracy---------------")
    accu = vis_utils.compare_accuracy_simple("excellent_unoriented", 1, path_init)
    show_accu(accu, "rruff (p)", print_info=True)

    print("-----------------Organic Raw accuracy-------------")
    accu_organic = vis_utils.compare_accuracy_simple("organic_target_raw", 1, path_init)
    show_accu(accu_organic, "organic (r)", print_info=True)

    print("-----------------Organic Preprocessed accuracy-------------")
    accu_organic = vis_utils.compare_accuracy_simple("organic_target", 1, path_init)
    show_accu(accu_organic, "organic (p)", print_info=True)

    print("------------------Bacteria accuracy-----------------------")
    accu_bacteria = vis_utils.compare_accuracy_simple("bacteria_reference_finetune", 1, path_init)
    show_accu(accu_bacteria, "Bacteria", print_info=True)


if __name__ == '__main__':
    args = give_args()
    save = args.save
    pdf_pgf = args.pdf_pgf
    path_init = args.dir2read_exp
    data_path = args.dir2read_data
    tds_dir = args.dir2save
    if not os.path.exists(tds_dir):
        os.makedirs(tds_dir)
    if args.index == "figure_augmentation_example":
        give_data_augmentation_example(tds_dir, save, pdf_pgf, data_path=data_path)
    elif args.index == "figure_example_spectra":
        show_example_spectra(tds_dir, save, pdf_pgf, data_path=data_path)
    elif args.index == "figure_uncertainty":
        give_uncertainty_distribution_figure_with_confidence_interval(tds_dir, save, pdf_pgf, path_init=path_init,
                                                                      data_path=data_path)
    elif args.index == "figure_conformal_bacteria":
        give_conformal_prediction_for_bacteria_paper(path_init=path_init, use_original_weight="original",
                                                     tds_dir=tds_dir, save=save, pdf_pgf=pdf_pgf,
                                                     data_path=data_path)
    elif args.index == "figure_motivation_conformal_prediction":
        motivation_for_conformal_prediction_multiple_datasets(save=save, pdf_pgf=pdf_pgf,
                                                              path_init=path_init,
                                                              path2save=tds_dir,
                                                              data_path=data_path)
    elif args.index == "conformal_prediction_correlation":
        give_conformal_prediction_for_multiple_datasets(path_init=path_init,
                                                        use_original_weight="weighted",
                                                        tds_dir=tds_dir, save=save, pdf_pgf=pdf_pgf,
                                                        data_path=data_path)
    elif args.index == "qualitative_results":
        give_qualitative_result_allinone(path_init, tds_dir=tds_dir, save=save, pdf_pgf=pdf_pgf,
                                         data_path=data_path)
    elif args.index == "table_overall_performance":
        report_accuracy_table(path_init=path_init)
