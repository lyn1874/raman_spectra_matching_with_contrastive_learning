"""
Created on 16:25 at 13/10/2021
@author: bo
Test the siamese network
"""
import sys
import torch
import numpy as np
import os
import model.model_inception as model_inception
import data.prepare_data as pdd
import vis_utils as vu


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


try:
    free_id = get_freer_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % get_freer_gpu()
except:
    print("GPU doesn't exist")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_model_baseon_modeldir(args, model_dir, data_group, print_info=True):
    """Give model and executable function based on the model directory
     Args:
         args: argument
         model_dir: directory to load ckpt
         data_group: [tr_data, tt_data, label_name]
         print_info: bool variable True/False
    """
    stem_max_dim=64
    depth=128
    [tr_spectrum, tr_label], [tt_spectrum, tt_label], label_name = data_group
    wavenumber = np.shape(tr_spectrum)[1]
    stem_kernel = 21
    distance_aggregate = "wave_channel_dot_L1"
    sim_model = get_model_inception(wavenumber, distance_aggregate, model_dir,
                                    stem_kernel=stem_kernel, stem_max_dim=stem_max_dim,
                                    depth=depth, print_model=print_info)
    testfunc = Test(args, sim_model, [tr_spectrum, tt_spectrum], [tr_label, tt_label], label_name)
    prediction, accu = testfunc.get_prediction_within_batch(tt_spectrum, tt_label, print_info=print_info)
    del testfunc.reference_features
    return prediction, accu


def get_model_inception(wavenumber, distance_aggregation, model_dir,
                        stem_kernel=3, stem_max_dim=64, depth=128, print_model=False):
    """Get the InceptionNet based feature extractor
    Note: at test time, I don't apply dropout operations anymore
    Args:
        wavenumber: int, number of wavenumber
        distance_aggregation: str, "wave_channel_dot_L1"
        model_dir: str, the directory to load the model
        stem_kernel: int
        stem_max_dim: int
        depth: int
        print_model: bool variable
    """
    sim_model = model_inception.Siamese(wavenumber,
                                        distance_aggregation,
                                        stem_kernel, depth=depth, stem_max_dim=stem_max_dim,
                                        dropout=False, within_dropout=False,
                                        separable_act=True)
    ckpt_dir = [model_dir + "/" + v for v in sorted(os.listdir(model_dir)) if ".pt" in v][-1]
    if print_model:
        print("load weight from", ckpt_dir)
    sim_model.load_state_dict(torch.load(ckpt_dir, map_location=device))
    sim_model.requires_grad_(False)
    sim_model.to(device)
    sim_model.eval()
    return sim_model


def get_data(args, model_dir, read_twin_triple="twin", print_info=True, dir2read="../data_group/"):
    [tr_spectrum, tr_label], [val_spectrum, val_label], \
    [tt_spectrum, tt_label], \
    pair_index, \
    label_name = pdd.read_dataset(args["raman_type"], model_dir, args["min_wave"],
                                  args["max_wave"], args["tot_num_per_mineral"],
                                  norm_first=args["norm_first"], norm_method=args["norm_method"],
                                  pre_define_tt_filenames=args["pre_define_tt_filenames"], min_freq=args["min_freq"],
                                  read_twin_triple=read_twin_triple,
                                  print_info=print_info, beta=1.0, data_path=dir2read)
    if "bacteria" not in args["raman_type"]:
        return [tr_spectrum, tr_label], [tt_spectrum, tt_label], \
               pair_index, label_name
    else:
        return [tr_spectrum, tr_label], [val_spectrum, val_label], [tt_spectrum, tt_label], \
                pair_index, label_name


def give_top_k_accu(tr_label, tt_label, pred, k, print_info=True):
    accu = np.zeros([k - 1])
    pred_score, _ = reorganize_similarity_score(pred, tr_label)
    for i in range(k)[1:]:
        pred_label = np.argsort(pred_score, axis=-1)[:, -i:]
        correct = [1 for j, v in enumerate(tt_label) if v in pred_label[j]]
        if print_info:
            print("Top %d Accuracy %.2f " % (i, (np.sum(correct) / len(tt_label)) * 100))
        accu[i - 1] = np.sum(correct) / len(tt_label) * 100
    return accu


class Test(object):
    def __init__(self, args, model, data, label, label_name):
        super(Test, self).__init__()
        self.dataset = args["dataset"]
        self.device = device
        self.sim_model = model
        self.num_wavelength = len(data[0][0])
        self.input_shape = [1, self.num_wavelength]
        tr_spectrum, tt_spectrum = data
        tr_label, tt_label = label
        self.tr_spectrum = self.dataloader(tr_spectrum, len(tr_spectrum))
        self.reference_features = self.sim_model.forward_test(self.tr_spectrum, [])
        del self.tr_spectrum
        self.tt_spectrum = tt_spectrum
        self.tr_label = tr_label.astype('int32')
        self.tt_label = tt_label.astype('int32')
        self.label_name = np.array(label_name)
        self.number_class = len(np.unique(self.label_name))

    def dataloader(self, spectrum, batch_size):
        """Read the spectrum in the torch tensor shape
        Args:
            spectrum: [batch_size, wavelength]
            batch_size: int
        """
        spectrum_tensor = torch.from_numpy(spectrum).to(torch.float32).unsqueeze(1).view(
            batch_size, 1, self.num_wavelength).to(self.device)
        return spectrum_tensor

    def get_prediction_within_batch(self, spectrum, tt_label, print_info=True):
        num = len(spectrum)
        batch_size = [i for i in range(50)[10:] if num % i == 0]
        batch_size = [batch_size[-1] if len(batch_size) > 1 else 50][0]
        num_iter = int(np.ceil(num / batch_size))
        prediction = []
        with torch.no_grad():
            for i in range(num_iter):
                if (i + 1) * batch_size < num:
                    s_spec = spectrum[i * batch_size:(i + 1) * batch_size]
                else:
                    s_spec = spectrum[i * batch_size:]
                    batch_size = len(s_spec)
                s_spec = self.dataloader(s_spec, batch_size)
                _prob = self.sim_model.forward_test_batch(s_spec, self.reference_features)
                prediction.append(_prob.detach().cpu().numpy())
        prediction = np.array([v for q in prediction for v in q])
        accuracy = give_top_k_accu(self.tr_label, tt_label, prediction[:len(tt_label)], 6,
                                   print_info=print_info)
        del s_spec
        return prediction, accuracy


def add_ensemble(prediction_g, string_use, norm_ensemble, group_index, tr_label, tt_label):
    if type(prediction_g[string_use[0]]) is list:
        prediction_avg = [np.zeros_like(prediction_g[string_use[0]][0]) for _ in prediction_g]
        for key in string_use:
            _predict_value = prediction_g[key]
            for i, s_pred in enumerate(_predict_value):
                prediction_avg[i] += s_pred
        prediction_g["ensemble_avg"] = [s_prediction / len(string_use) for s_prediction in prediction_avg]
    else:
        prediction_avg = np.zeros(prediction_g[string_use[0]].shape)
        for key in string_use:
            prediction_avg += prediction_g[key]
        prediction_g["ensemble_avg"] = prediction_avg / len(string_use)
    if norm_ensemble:
        prediction_g["ensemble_avg"] = vu.weight_prediction_fast(prediction_g["ensemble_avg"],
                                                                 group_index, tr_label, tt_label)
    else:
        correct = np.sum([tr_label[np.argmax(prediction_avg / len(string_use), axis=-1)] == tt_label]) / len(tt_label)
        # print("The accuracy on using ensembled prediction is %.2f" % (correct * 100))
    string_out = np.concatenate([string_use, ["ensemble_avg"]], axis=0)
    return prediction_g, string_out


def reorganize_similarity_score(prediction, tr_label):
    """Reorganize the similarity score -> [num_test_sample, num_classes]
    Args:
        prediction: [num_test_sample, num_training_sample]
        tr_label: [num_training_sample]
        tt_label: [num_test_sample]
        top_k: integer
    """
    if type(prediction) == list:
        prediction = np.array(prediction)
    index = [np.where(tr_label == i)[0] for i in np.unique(tr_label)]
    prediction_baseon_class = [np.max(prediction[:, v], axis=-1) for v in index]
    correspond_tr_index = [v[np.argmax(prediction[:, v], axis=-1)] for v in index]
    prediction_baseon_class = np.array(np.transpose(prediction_baseon_class, (1, 0)))
    correspond_tr_index = np.array(np.transpose(correspond_tr_index, (1, 0)))
    return prediction_baseon_class, correspond_tr_index


def get_conformal_prediction_threshold(tr_data, val_data, label_name, args, ckpt_dir):
    if "bacteria" not in args["raman_type"]:
        reference_val_data, val_data = pdd.get_fake_reference_and_test_data(tr_data, 1, data=args["dataset"])
    else:
        reference_val_data = tr_data
    string_val_use = ["ensemble_%d" % i for i in range(len(ckpt_dir))]
    prediction_val_g = {}
    for i, s_ckpt in enumerate(ckpt_dir):
        prediction, _ = get_model_baseon_modeldir(args, s_ckpt,
                                                  [reference_val_data, val_data, label_name],
                                                  print_info=False)
        prediction_val_g[string_val_use[i]] = prediction
    group_index = [np.where(reference_val_data[1] == i)[0] for i in np.unique(val_data[1])]
    prediction_val_g, string_val_use = add_ensemble(prediction_val_g, string_val_use,
                                                         False, group_index,
                                                         reference_val_data[1], val_data[1])
    val_prediction_baseon_cls, _ = reorganize_similarity_score(prediction_val_g[string_val_use[-1]],
                                                                    reference_val_data[1])
    return val_prediction_baseon_cls, reference_val_data[1]

