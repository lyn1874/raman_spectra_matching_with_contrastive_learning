"""
Created on 13:59 at 15/10/2021/
@author: bo
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter, NullFormatter
import os
import numpy as np
import math
from scipy.special import softmax


def give_avg_std(perf, print_info=False):
    perf = np.array(perf)
    avg = np.mean(perf, axis=0)
    std = np.std(perf, axis=0) * 1.95 / np.sqrt(len(perf))
    if print_info:
        for i in range(5):
            print("Top %d, accuracy %.2f with 95 confidence interval %.2f" % (i + 1, avg[i, 1],
                                                                              std[i, 1]))
    return avg, std


def weight_prediction_fast(prediction_original, group_index, tr_label=[], tt_label=[]):
    prediction = prediction_original.copy()
    prediction = softmax(prediction, axis=-1)
    upper_bound = [upper_bound_multiple_spectra(prediction[:, v]) for v in group_index]
    upper_bound = np.transpose(upper_bound, (1, 0))
    weights = upper_bound / np.max(upper_bound, axis=-1, keepdims=True)
    weights_repeat = [np.repeat(weights[:, i:i+1], len(q), axis=-1) for i, q in enumerate(group_index)]
    weights = np.concatenate(weights_repeat, axis=-1)
    prediction = prediction * weights
    if len(tr_label) > 0:
        original_accu = np.sum(tr_label[np.argmax(prediction_original, axis=-1)] == tt_label) / len(tt_label) * 100
        updated_accu = np.sum(tr_label[np.argmax(prediction, axis=-1)] == tt_label) / len(tt_label) * 100
        print("original accuracy: %.2f vs updated accuracy: %.2f" % (original_accu, updated_accu))
    return prediction, [original_accu, updated_accu]


def majority_voting(prediction, tr_label, tt_label, num):
    argsort_index = np.argsort(prediction, axis=-1)[:, ::-1]
    predict_label = [tr_label[v[:num]] for v in argsort_index]
    predict_single_label = np.zeros_like(tt_label)
    for i, s_pred in enumerate(predict_label):
        _u_s, _u_c = np.unique(s_pred, return_counts=True)
        predict_single_label[i] = _u_s[np.argmax(_u_c)]
    accuracy = np.sum([v == q for v, q in zip(predict_single_label, tt_label)]) / len(tt_label)
    return accuracy, predict_single_label


def enlarge_feature_maps(feature_map, factor=4):
    """Enlarge feature map by a factor"""
    new_feat = np.zeros([len(feature_map) * factor, np.shape(feature_map)[1]])
    for i in range(len(feature_map)):
        new_feat[i*factor:(i+1)*factor] = feature_map[i:i+1]
    return new_feat


def extract_accuracy_from_txt_file_simple(raman_type, path_init):
    path2read = path_init + "/" + raman_type + "/tds/"
    txt_file_group = [v for v in sorted(os.listdir(path2read)) if '.txt' in v]
    output = []
    for txt_file in txt_file_group:
        txt_file = path2read + txt_file
        with open(txt_file, 'r') as f:
            content = f.readlines()
            performance = [v for v in content if "The accuracy on using ensembled prediction" in v]
            performance = [float(v.split("is ")[-1].split("\n")[0]) for v in performance]
            ensemble_perf = np.reshape(performance, [-1, 4])
            single_performance = [v for v in content if "original accuracy" in v and "updated accuracy" in v]
            single_original_perf = [float(v.split(": ")[1].split(" vs")[0]) for v in single_performance]
            single_updated_perf = [float(v.split("updated accuracy: ")[1].split("\n")[0]) for v in single_performance]
            single_performance = np.concatenate([np.reshape(single_original_perf, [-1, 1]),
                                                 np.reshape(single_updated_perf, [-1, 1])], axis=1)[
                                 len(single_original_perf) // 2:]

            counting_performance = [v for v in content if "Count" in v and "accuracy" in v]
            counting_performance = np.reshape(counting_performance, [-1, 6])
            counting_accuracy = np.zeros([len(counting_performance), 4])
            counting_count = np.zeros([len(counting_performance), 4])
            for i in range(len(counting_performance)):
                temp = [v.split(" ") for v in counting_performance[i]]
                _count_stat = [int(v[1].split(":")[0]) for v in temp]
                _accu_stat = [float(v[-1].split("\n")[0]) for v in temp]
                counting_count[i, :] = [_count_stat[j] for j in [1, 2, 4, 5]]
                counting_accuracy[i, :] = [_accu_stat[j] for j in [1, 2, 4, 5]]
        output.append([single_performance, ensemble_perf, counting_accuracy, counting_count])
    return output


def show_top5_prediction(select_index, prediction, matched_index, data_group, wavenumber, top5_index=[]):
    norm_score = softmax(prediction, axis=-1)[select_index]
    if len(top5_index) == 0:
        top5_index = np.argsort(norm_score)[-5:][::-1]
    matched_top5 = matched_index[select_index][top5_index]
    [tr_spectra, tr_label], [tt_spectra, tt_label], label_name = data_group
    if "class" in label_name[0]:
        label_name = [v.replace("class", "organic") for v in label_name]
    legend_g = ["Input:%s" % label_name[tt_label[select_index]]]
    for i in range(len(top5_index)):
        legend_g.append("Top-%d: %s (%.2f)" % (i+1, label_name[top5_index[i]], (norm_score[top5_index[i]]*100)))
    fig = plt.figure(figsize=(7, 6))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for i in range(len(top5_index)+1):
        if i == 0:
            c_u = 'r'
        else:
            if top5_index[i-1] == tt_label[select_index]:
                c_u = 'r'
            else:
                c_u = np.random.random([3])
        ax = fig.add_subplot(len(top5_index)+1, 1, i+1)
        if i == 0:
            ax.plot(wavenumber, tt_spectra[select_index], color=c_u, label=legend_g[i])
        else:
            ax.plot(wavenumber, tr_spectra[matched_top5[i-1]], color=c_u, label=legend_g[i])
        ax.legend(loc='best')
        if i != len(top5_index):
            ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax_global.set_xlabel("\nWavenumber (cm" + r"$^{-1}$" + ")")
    ax_global.set_ylabel("Intensity (a.u.) \n")



def compare_accuracy_simple(raman_type, select_step, path_init):
    output = extract_accuracy_from_txt_file_simple(raman_type, path_init)
    stat_single_step = []
    for i in range(4):
        _stat = [v[-2][select_step, i] for v in output]  # 0.05, 0.1, 0.5, 2
        stat_single_step.append(_stat)
    return np.array(stat_single_step)


def upper_bound_multiple_spectra(predictions):
    q1 = np.quantile(predictions, 0.25, axis=-1)
    q3 = np.quantile(predictions, 0.75, axis=-1)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return upper_bound


def show_twinx(alpha_group, accuracy, set_size, ax):
    if not ax:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        ax.set_ylabel("Empirical coverage", color='r')
        ax.set_xlabel("Theoretical coverage (1 -" + r'$\alpha$' + ")")
        ax_t = ax.twinx()
        ax_t.set_ylabel("Average set size", color='g')
    else:
        ax_t = ax.twinx()
    scale = [100 if np.max(alpha_group) > 1 else 1][0]
    ax.plot(scale - alpha_group, accuracy, color='r', marker='.', label="Accuracy")
    ax.tick_params(axis='y', labelcolor='r')
    if np.max(accuracy) > 1:
        ax.set_ylim(top=100.0)
    else:
        ax.set_ylim(top=1.0)
    ax_t.plot(scale - alpha_group, set_size, color='g',  marker='.', label="Average set size")
    ax_t.tick_params(axis='y', labelcolor='g')
    ax_t.set_yscale("symlog")
    ax_t.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    ax_t.yaxis.set_minor_formatter(NullFormatter())
    ax_t.grid(False)


def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    return ax_global

