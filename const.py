"""
Created on 12:46 at 02/03/2021
@author: bo 
"""
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    """This function is used to give the argument"""
    parser = argparse.ArgumentParser(description='Beta-VAE')
    parser.add_argument('-ds', '--dataset', type=str, default="RRUFF", metavar='RRUFF',
                        help='dataset')
    parser.add_argument('--raman_type', type=str, default="excellent_unoriented",
                        help="which data type do I want to use? excellent_unoriented, or lr_raman")
    parser.add_argument('--augment_option', type=str, default="none",
                        help="which augmentation method do I want to use?")
    parser.add_argument('-bs', '--batch_size', type=int, default=30, metavar='BATCH_SIZE',
                        help='input batch size for training (default: 100)')
    parser.add_argument('-ep', '--max_epoch', type=int, default=100, metavar='EPOCHS',
                        help='maximum number of epochs')
    parser.add_argument('--min_wave', type=int, default=196, metavar='MIN_WAVE',
                        help='the smallest wavenumber')
    parser.add_argument('--max_wave', type=int, default=1404, metavar='MAX_WAVE',
                        help='the largest wavenumber')  # the max wavenumber for raw data is actually 1368
    parser.add_argument("--min_freq", type=int, default=2, metavar="MIN_FREQ",
                        help="the minimum number of spectrum for per mineral")
    parser.add_argument('--tot_num_per_mineral', type=int, default=0, metavar="TOT_NUM_PER_MINERAL",
                        help='the total number of spectrum for each mineral')
    parser.add_argument('--norm_method', type=str, default="max_1_min_0")
    parser.add_argument('--norm_first', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('-lg', '--learning_rate_init', type=float, default=0.001, metavar='LRATE_G_INIT',
                        help='the initial learning rate')
    parser.add_argument('--version', type=int, default=0, help="experiment version")
    parser.add_argument('--lower_bound', type=int, default=10, help="the lower bound")  # 8
    parser.add_argument('--upper_bound', type=int, default=1280, help="the upper bound")  # 1480
    parser.add_argument("--rand_shift", type=int, default=2.5, help="random shiftting")
    parser.add_argument("--sliding_window", type=int, default=30, help="the sliding window for calculating std")
    parser.add_argument("--aug_more_than_one", type=str2bool, default=False,
                        help="whether augment the minerals that have more than one spectrums during sampling")
    parser.add_argument("--noise_std", type=int, default=1, help="std for the added gaussian noise")
    parser.add_argument("--augmentation_mode", type=str, default="history",
                        help="The augmentation methods for the minerals that only have spectrum in the training data")
    parser.add_argument("--alpha", type=float, default=0,
                        help="the ratio between the number of positive pairs and negative pairs")
    parser.add_argument("--beta", type=float, default=1.0, help="whether a negative pair is selected or not")
    parser.add_argument("--read_twin_triple", type=str, default="twin",
                        help="how to real the pairs index, twin, twin_hard, triple")
    parser.add_argument("--neg_shift_mean", type=int, default=30,
                        help="the mean the shift that is sampled from a Gaussian distribution")
    parser.add_argument("--neg_shift_scale", type=int, default=10,
                        help="how many wavenumbers need to be shifted in order to create hard negative samples")
    parser.add_argument("--pos_shift_mean", type=int, default=4)
    parser.add_argument("--pos_shift_std", type=int, default=1)
    parser.add_argument("--pos_shift_scale", type=int, default=1)
    parser.add_argument("--distance_aggregation", type=str, default="wave_channel_dot_L1")
    parser.add_argument("--stem_kernel", type=int, default=3)
    parser.add_argument("--balanced", type=str2bool, default=True)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--l2_regu_para", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--dropout", type=str2bool, default=False)
    parser.add_argument("--stem_max_dim", type=int, default=64)
    parser.add_argument("--within_dropout", type=str2bool, default=False)
    parser.add_argument("--siamese_version", type=int, default=4)
    parser.add_argument("--random_leave_one_out", default=False, type=str2bool)
    parser.add_argument("--separable_act", default=False, type=str2bool)
    parser.add_argument("--check_augmentation_on_datasplit", default=False, type=str2bool)
    parser.add_argument("--check_distance_on_datasplit", default=False, type=str2bool)
    parser.add_argument("--repeat_on_python", default=False, type=str2bool)
    parser.add_argument("--repeat_time", default=0, type=int)
    parser.add_argument("--s_augmentation", default="none", type=str)
    parser.add_argument("--s_distance", default="wave_channel_dot_product", type=str)
    parser.add_argument("--check_ratio_on_datasplit", default=False, type=str2bool)
    parser.add_argument("--s_ratio", default=0.05, type=float)
    parser.add_argument("--dir2save_ckpt", type=str, help="The directory to save the ckpt files")
    parser.add_argument("--dir2load_data", type=str, help="The directory to load the data from ")
    return parser.parse_args()


def give_args_test(raman_type="excellent_unoriented"):
    param = {}
    param["raman_type"] = raman_type
    param["noise_std"] = 5
    param["lower_bound"] = 0
    param["upper_bound"] = 0
    param["min_freq"] = 2
    param["num_encoder_layer"] = 4
    param["rand_shift"] = 3

    if raman_type == "excellent_unoriented":
        param["min_wave"] = 76
        param["max_wave"] = 1404
        param["norm_method"] = "max_1_min_0"
        param["dataset"] = "RRUFF"
    elif raman_type == "raw":
        param["min_wave"] = 40
        param["max_wave"] = 1496
        param["norm_method"] = "max_1_min_0"
        param["dataset"] = "RRUFF"
        param["noise_std"] = 3
        param["rand_shift"] = 0
    elif raman_type == "organic_target" or raman_type == "organic_target_raw":
        param["min_wave"] = 0
        param["max_wave"] = 1024
        param["norm_method"] = "abs"
        param["dataset"] = "ORGANIC"
        param["noise_std"] = 1
    elif "bacteria" in raman_type:
        param["min_wave"] = 0
        param["max_wave"] = 1000
        param["norm_method"] = "none"
        param["dataset"] = "BACTERIA"
        param["noise_std"] = 0
    param["tot_num_per_mineral"] = 0
    param["norm_first"] = False
    param["augment_option"] = "none"
    return param

