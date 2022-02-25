"""
Created on 13:07 at 09/07/2021
@author: bo 
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import model.model_inception as model_inception
import data.rruff as rruff
import const
import data.prepare_data as pdd
import data.bacteria as bacteria
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import sys


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


class Train(object):
    def __init__(self, args, model_dir, dir2save_data=None, pre_define_tt_filenames=False,
                 dir2load_data="../data_group/"):
        """Arguments:
        args: Arguments per dataset
        model_dir: str, a directory to save the model checkpoints
        dir2save_data: str, a directory to save the training/validation/test data
        pre_define_tt_filenames: bool, whether to load the saved training/validation/test dataset or to re-initialise
            the training/validation/testing dataset
        dir2load_data: str, a directory to load the dataset from
        """
        super(Train, self).__init__()
        self.dataset = args.dataset
        self.raman_type = args.raman_type
        self.model_dir = model_dir
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.device = device
        self.num_workers = args.num_workers
        self.min_wave = args.min_wave
        self.max_wave = args.max_wave
        self.num_wavelength = self.max_wave - self.min_wave
        self.tot_num_per_mineral = args.tot_num_per_mineral
        self.augment_option = args.augment_option
        self.lower_upper_bound = [args.lower_bound, args.upper_bound]
        self.rand_shift = args.rand_shift
        self.sliding_window = args.sliding_window
        self.input_shape = [1, self.num_wavelength]
        self.noise_std = args.noise_std
        self.norm_first = args.norm_first
        self.norm_method = args.norm_method
        self.augmentation_mode_on_single_spectrum = args.augmentation_mode
        self.read_twin_triple = args.read_twin_triple
        self.beta = args.beta
        self.distance_aggregation = args.distance_aggregation
        self.pos_shift_params = [args.pos_shift_scale, args.pos_shift_std, args.pos_shift_mean]
        self.neg_shift_params = [args.neg_shift_scale, 10, args.neg_shift_mean]
        self.stem_kernel = args.stem_kernel
        self.stem_max_dim = args.stem_max_dim
        self.channel_depth = args.depth
        self.lr_schedule = args.lr_schedule
        self.l2_regu_para = args.l2_regu_para
        self.balanced = args.balanced
        self.balanced_alpha = args.alpha
        self.dropout = args.dropout
        self.within_dropout = args.within_dropout
        self.separable_act = args.separable_act
        if not dir2save_data:
            self.dir2save_data = self.model_dir
        else:
            self.dir2save_data = dir2save_data
        self.dir2load_data = dir2load_data
        self.pre_define_tt_filenames = pre_define_tt_filenames
        self.random_leave_one_out = args.random_leave_one_out

        a = torch.from_numpy(np.zeros([1])).to(self.device)
        stdoutOrigin = sys.stdout
        sys.stdout = open(self.model_dir + "training_statistics.txt", 'w')

        [tr_spectrum, tr_label], [val_spectrum, val_label], \
        [tt_spectrum, tt_label], \
        [positive_pair, negative_pair], \
        label_name = pdd.read_dataset(self.raman_type, self.dir2save_data,
                                      self.min_wave, self.max_wave, self.tot_num_per_mineral,
                                      self.norm_first, self.norm_method,
                                      self.lower_upper_bound[0], self.lower_upper_bound[1],
                                      self.sliding_window, self.rand_shift, self.noise_std,
                                      pre_define_tt_filenames=self.pre_define_tt_filenames, min_freq=2,
                                      read_twin_triple=self.read_twin_triple, beta=self.beta, show=False,
                                      random_leave_one_out=self.random_leave_one_out,
                                      augmentation_mode_for_single_spectrum=self.augmentation_mode_on_single_spectrum,
                                      data_path=self.dir2load_data)
        sim_model = model_inception.Siamese(self.num_wavelength,
                                            self.distance_aggregation,
                                            stem_kernel=self.stem_kernel,
                                            depth=self.channel_depth,
                                            stem_max_dim=self.stem_max_dim,
                                            dropout=self.dropout,
                                            within_dropout=self.within_dropout,
                                            separable_act=self.separable_act)
        self.sim_model = sim_model.to(device)
        self.sim_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.label_name = np.array(label_name)

        transform = pdd.give_transformation(self.augment_option, self.pos_shift_params)
        positive_loader = pdd.MineralDataLoad(tr_spectrum, tr_label, positive_pair, transform)
        negative_loader = pdd.MineralDataLoad(tr_spectrum, tr_label, negative_pair, transform)

        pos_count, neg_count = len(positive_pair), len(negative_pair)
        self.neg_count = np.sum(negative_pair[:, -2])
        pos_neg_label = np.zeros([pos_count + neg_count])
        pos_neg_label[:len(positive_pair)] = pos_count
        pos_neg_label[len(positive_pair):] = neg_count
        samples_weight = (pos_count + neg_count) / pos_neg_label
        print("The balancing alpha", self.balanced_alpha)
        if self.balanced_alpha != 0:
            samples_weight[:len(positive_pair)] = samples_weight[:len(positive_pair)] * self.balanced_alpha
            print("Manipulating the ratio between positive and negative pair", self.balanced_alpha)
        samples_weight = torch.from_numpy(samples_weight)
        print("-------------------------------------------------------------------------------------")
        print("The weight for the samples", samples_weight.unique())
        if self.balanced or self.balanced_alpha != 0:
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        else:
            print("----------There is no balancing---------------")
        concat_dataset = ConcatDataset([positive_loader, negative_loader])
        sys.stdout.close()
        sys.stdout = stdoutOrigin

        num_class = len(tt_label)
        self.num_classes = num_class
        if self.dataset == "RRUFF":
            if self.raman_type == "excellent_unoriented":
                self.fix_number_iteration = 8326
            elif self.raman_type == "raw":
                self.fix_number_iteration = 7673
                if self.augmentation_mode_on_single_spectrum == "none":
                    self.fix_number_iteration = 2841
                else:
                    self.fix_number_iteration = 7673
        elif self.dataset == "ORGANIC" and "organic_target" in self.raman_type:
            if self.augment_option == "none":
                self.fix_number_iteration = (pos_count + neg_count) // self.batch_size
            elif self.augment_option == "sample":
                if self.tot_num_per_mineral == 6:
                    self.fix_number_iteration = (pos_count + neg_count) // self.batch_size // 3
                else:
                    self.fix_number_iteration = 81 * 7
        elif self.dataset == "BACTERIA":
            if self.raman_type == "bacteria_reference_finetune" \
                        or self.raman_type == "bacteria_random_reference_finetune":
                self.fix_number_iteration = 9903  # 17998
        train_params = {"batch_size": self.batch_size,
                        "drop_last": True,
                        "num_workers": self.num_workers,
                        "pin_memory": True,
                        "worker_init_fn": lambda _: np.random.seed(),
                        }
        if self.balanced:
            print("Balance my negative and positive pair")
            train_params.update({"sampler": sampler})
        else:
            print("I need to shuffle my training dataset")
            train_params.update({"shuffle": True})

        tr_data_loader = DataLoader(concat_dataset, **train_params)

        # ----- Here is for the validation dataset --------- #
        validation_positive_pair, validation_negative_pair = rruff.find_manu_pos_neg_pair(tr_label,
                                                                                          np.ones([len(tr_label)]),
                                                                                          beta=0.0)
        validation_transform_pos_hard = pdd.give_transformation(self.augment_option, self.pos_shift_params)
        validation_transform_neg_hard = pdd.give_neg_transformation(self.neg_shift_params)
        validation_positive_loader = pdd.MineralDataLoadHardManu(tr_spectrum, tr_label, validation_positive_pair,
                                                                 validation_transform_pos_hard, None)
        validation_negative_loader = pdd.MineralDataLoadHardManu(tr_spectrum, tr_label, validation_negative_pair,
                                                                 validation_transform_pos_hard,
                                                                 validation_transform_neg_hard)
        validation_concat_dataset = ConcatDataset([validation_positive_loader,
                                                   validation_negative_loader])
        val_params = {"batch_size": self.batch_size,
                      "drop_last": True,
                      "num_workers": self.num_workers,
                      "pin_memory": True,
                      "worker_init_fn": lambda _: np.random.seed(),
                      "shuffle": False}
        val_data_loader = DataLoader(validation_concat_dataset, **val_params)

        self.model_dir = model_dir

        train_logdir = self.model_dir.split("version")[0] + "tb/" + self.model_dir.split("version_")[1]
        train_logdir = train_logdir + "/logs/"
        if not os.path.exists(train_logdir):
            os.makedirs(train_logdir)
        self.train_logdir = train_logdir
        self.train_writer = SummaryWriter(log_dir=train_logdir)

        parameter_list = [p for p in self.sim_model.parameters() if p.requires_grad]
        if self.l2_regu_para == 0:
            optimizer = optim.Adam(parameter_list, lr=args.learning_rate_init,
                                   eps=1e-4)
        else:
            optimizer = optim.Adam(parameter_list, lr=args.learning_rate_init,
                                   eps=1e-4, weight_decay=self.l2_regu_para)
        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epoch * len(tr_data_loader) // self.fix_number_iteration)
        elif self.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        else:
            print("-----The predefined lr schedule doesn't exist----")
            scheduler = None

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_loader = tr_data_loader
        self.val_data_loader = val_data_loader
        del a
        # --- Here is the test data --------- #
        if self.raman_type != "bacteria_reference":
            self.reference_spectrum_tensor = pdd.load_test_data(tr_spectrum, self.num_wavelength, device)
            reference_label = tr_label
        else:
            _, finetune_data, _, _ = bacteria.get_reference_data("../bacteria/", False, None, False)
            self.reference_spectrum_tensor = pdd.load_test_data(finetune_data[0], self.num_wavelength, device)
            reference_label = finetune_data[1]
        
        self.test_spectrum_tensor = pdd.load_test_data(tt_spectrum, self.num_wavelength, device)

        # ---- Create the paired label at test time ---- #
        tr_label_exp = np.repeat(np.expand_dims(reference_label, axis=0), repeats=len(tt_label), axis=0)
        tr_label_exp = tr_label_exp - np.expand_dims(tt_label, axis=1)
        self.pair_label_test = (tr_label_exp == 0).astype(np.int32)

        # --- Here is the accuracy on the validation dataset because now I have some real validation dataset --- #
        if len(val_spectrum) > 0:
            self.validation_spectrum_tensor = pdd.load_test_data(val_spectrum, self.num_wavelength, device)
            val_label_exp = np.repeat(np.expand_dims(reference_label, axis=0), repeats=len(val_label), axis=0)
            val_label_exp = val_label_exp - np.expand_dims(val_label, axis=1)
            self.pair_label_val = (val_label_exp == 0).astype(np.int32)
            self.val_label = val_label
        else:
            self.pair_label_val = []
            self.val_label = []

        self.tr_label = tr_label
        self.tt_label = tt_label
        print("Tr label shape", np.shape(self.tr_label))
        print("Tt label shape", np.shape(self.tt_label))
        print("Paired label shape", np.shape(self.pair_label_test))
        self.metrics = {}
        print("dataset:", self.dataset)
        print("imshape:", self.input_shape)
        print("number of classes", self.num_classes)
        print("initial lr : %.5f" % args.learning_rate_init)
        print("tr iteration :", len(self.train_data_loader))
        print("The actual number of epoch", self.max_epoch * len(self.train_data_loader) / self.fix_number_iteration)
        print("The number of fake epochs in per epoch", len(self.train_data_loader) / self.fix_number_iteration)
        print("--------------------------------------------")

    def _trans_data(self, _spec, _label):
        _spec = _spec.permute(2, 0, 1, 3).reshape([self.batch_size * 2, 1, self.num_wavelength]).to(self.device)
        _label_cls = _label[:, :2].permute(1, 0).reshape(self.batch_size * 2).type(torch.LongTensor).to(self.device)
        _label_pair = _label[:, -1:].to(torch.float32).to(self.device)
        return _spec, _label_pair, _label_cls

    def _update_per_batch(self, spectrum, _label):
        """Update per batch
        Args:
            spectrum: [batch_size, 1, 2, self.num_wavelength] torch.float32
            _label: [batch_size, 2] torch.float32
        """
        _input_tr, _label_pair, _label_cls = self._trans_data(spectrum, _label)
        self.optimizer.zero_grad()
        _tr_pred = self.sim_model(_input_tr)
        _sim_loss = self.sim_loss(_tr_pred, _label_pair).div(self.batch_size)
        self.metrics["tr_siamese_loss"] += _sim_loss.data.item()
        _loss_back = _sim_loss
        self.metrics["positive_negative_ratio"] += _label_pair.sum().div((1.0 - _label_pair).sum()).data.item()
        _loss_back.backward()
        self.optimizer.step()

    def _eval_on_all_val(self, epoch):
        self.sim_model.eval()
        val_loss, val_fc_loss = 0, 0
        for batch_idx, (_spectrum, _label, _move) in enumerate(self.val_data_loader):
            _spectrum, _label_pair, _label_cls = self._trans_data(_spectrum, _label)
            _val_pred = self.sim_model(_spectrum)
            _sim_loss = self.sim_loss(_val_pred, _label_pair).div(self.batch_size)
            val_loss += _sim_loss.data.item()
        self.train_writer.add_scalar("model/val_siamese_loss",
                                     val_loss / len(self.val_data_loader), epoch)
        print("---Done with validation")
        return val_loss / len(self.val_data_loader)

    def _eval_on_all_test(self, epoch, input_tensor, input_pair_label, input_label, input_str):
        """Args: input_tensor: self.test_spectrum_tensor, self.test_pair_label"""
        self.sim_model.eval()
        print("--------Evaluation on the test data at epoch %d" % epoch)
        self.reference_features = self.sim_model.forward_test(self.reference_spectrum_tensor, [])
        print("The reference feature shape", self.reference_features.shape)
        num_batches = int(np.ceil(len(input_tensor) / self.batch_size))
        tt_loss, tt_accu = 0, 0
        tt_fc_loss, tt_fc_accu = 0, 0
        for i in range(num_batches):
            if i != num_batches - 1:
                _tt_spectrum = input_tensor[i * self.batch_size: (i + 1) * self.batch_size]
                _pair_label_npy = input_pair_label[i * self.batch_size: (i + 1) * self.batch_size]
            else:
                _tt_spectrum = input_tensor[i * self.batch_size:]  # [batch_size, 1, num_features]
                _pair_label_npy = input_pair_label[i * self.batch_size:]  # [batch_size, num_training_spectra]
            _distance = self.sim_model.forward_test_batch(_tt_spectrum, self.reference_features)
            _pair_label = torch.from_numpy(_pair_label_npy).to(torch.float32).to(self.device)
            _val_loss = self.sim_loss(_distance, _pair_label)
            # ----- Here is for the matrix based distance calculation --- #
            _, _pred_label = torch.max(_distance, 1)
            _pred_label = _pred_label.detach().cpu().numpy()
            _val_accuracy = np.sum([v[j] for v, j in zip(_pair_label, _pred_label)])
            tt_loss += _val_loss
            tt_accu += _val_accuracy
            # ----- Here is the for the vector based distance calculation --- #
        self.train_writer.add_scalar("model/%s_siamese_loss" % input_str,
                                     tt_loss / len(input_label) / len(self.reference_spectrum_tensor), epoch)
        self.train_writer.add_scalar("model/%s_siamese_accu" % input_str,
                                     tt_accu / len(input_label), epoch)
        print("epoch", epoch, "testing loss", tt_loss.data.item() / len(input_label),
              "testing accuracy", tt_accu / len(input_label))
        print("-------Done with testing ")
        del self.reference_features

    def _log_callback(self, epoch):
        for name, p in self.sim_model.named_parameters():
            self.train_writer.add_histogram(f"{name.replace('.', '/')}/value", p, epoch)
            self.train_writer.add_histogram(f"{name.replace('.', '/')}/grad", p, epoch)

    def _save_model(self, epoch):
        torch.save(self.sim_model.state_dict(), self.model_dir + "/model-{:04d}.pt".format(epoch))

    def _get_empty_metrics(self):
        self.metrics["tr_siamese_loss"] = 0.0
        self.metrics["shift"] = []
        self.metrics["positive_negative_ratio"] = 0.0

    def run(self):
        # warmup the batchnorm layers --> based on cedric's implementation
        tot_batch_per_train_loader = len(self.train_data_loader)
        self.sim_model.train()
        with torch.no_grad():
            for batch_idx, (_spectrum, _label, _move) in enumerate(self.train_data_loader):
                _spectrum, _label_pair, _label_cls = self._trans_data(_spectrum, _label)
                if batch_idx == 0:
                    print("spectrum shape", _spectrum.shape)
                    print("paired label shape", _label_pair.shape, _label_pair.max(), _label_pair.min())
                    print("classification label shape", _label_cls.shape, _label_cls.min(), _label_cls.max())
                    _la_diff = (_label_cls[:self.batch_size] - _label_cls[self.batch_size:] == 0).to(torch.float32)
                    if self.read_twin_triple == "twin":
                        print("paired label equal to cls label:", (_label_pair.squeeze(1) - _la_diff).sum(),
                              " should equal to 0")
                    print("the shifting scale", _move.shape, _move.max(), _move.min())
                _ = self.sim_model(_spectrum)
                if batch_idx > 1000:
                    print(".....will not loop over the whole dataset during warmup process")
                    break

        self._log_callback(0)
        global_step = 1  # changed Jun-1st to global_step = 0
        # global_step = 0
        with torch.no_grad():
            # self._eval_on_all_val(0)
            if len(self.val_label) > 0:
                self._eval_on_all_test(0, self.validation_spectrum_tensor,
                                       self.pair_label_val, self.val_label, "validation")
            self._eval_on_all_test(0, self.test_spectrum_tensor,
                                   self.pair_label_test, self.tt_label, "test")
        val_loss_group = []
        for epoch in range(1, self.max_epoch + 1):
            self.sim_model.train()
            self._get_empty_metrics()
            for idx, (_spectrum, _label, _move) in enumerate(tqdm(self.train_data_loader)):
                self._update_per_batch(_spectrum, _label)
                global_step += 1  # changed Jun-1st
                if global_step % self.fix_number_iteration == 0:
                    fake_epoch = global_step // self.fix_number_iteration
                    self.train_writer.add_scalar("model/tr_siamese_loss",
                                                 self.metrics["tr_siamese_loss"] / self.fix_number_iteration,
                                                 fake_epoch)
                    self.train_writer.add_scalar("model/positive_negative_ratio",
                                                 self.metrics["positive_negative_ratio"] / self.fix_number_iteration,
                                                 fake_epoch)
                    print("epoch", fake_epoch, "training loss %.2f" % (self.metrics["tr_siamese_loss"]))
                    with torch.no_grad():
                        if self.dataset != "BACTERIA":
                            _val_loss = self._eval_on_all_val(fake_epoch)
                        # val_loss_group.append(_val_loss)
                        if len(self.val_label) > 0:
                            self._eval_on_all_test(fake_epoch, self.validation_spectrum_tensor,
                                                   self.pair_label_val, self.val_label, "validation")
                        self._eval_on_all_test(fake_epoch, self.test_spectrum_tensor,
                                               self.pair_label_test, self.tt_label, "test")

                    self.sim_model.train()
                    if self.lr_schedule == "cosine":
                        self.scheduler.step()
                        current_lr = self.scheduler.get_lr()[0]
                    self.train_writer.add_scalar('model/learning_rate', current_lr,
                                                 fake_epoch)
                    self._get_empty_metrics()

                    # if val_loss_group[-1] <= np.mean(val_loss_group[-10:-1]) and fake_epoch >= 2:
                    if fake_epoch % 4 == 0 and fake_epoch >= 2:
                        self._save_model(fake_epoch)
                if global_step % (tot_batch_per_train_loader // 9) == 0:
                    fake_epoch = global_step // self.fix_number_iteration
                    self._save_model(fake_epoch)
                # global_step += 1  # changed Jun-1st
            if self.max_epoch - epoch < 30:
                if epoch % 1 == 0 or epoch == self.max_epoch:
                    self._save_model(global_step // self.fix_number_iteration)
        self.train_writer.close()

        # delete all the tensors that use the GPU
        del self.sim_model
        del self.train_data_loader
        del self.test_spectrum_tensor
        if "bacteria" in self.raman_type:
            del self.validation_spectrum_tensor
        torch.cuda.empty_cache()


def create_dir(dir_use):
    if not os.path.exists(dir_use):
        os.makedirs(dir_use)


if __name__ == '__main__':
    args = const.give_args()
    model_mom = args.dir2save_ckpt + "/%s/" % args.raman_type + "version_%d_" % args.version
    if args.raman_type == "excellent_unoriented":
        args.augment_option = "sample"
        args.sliding_window = 10
        args.tot_num_per_mineral = 6
        args.noise_std = 5
        args.rand_shift = 0
        args.min_wave = 76
        if args.augmentation_mode == "none":
            args.augment_option = "none"
            args.tot_num_per_mineral = 0
    if args.raman_type == "raw":
        args.augment_option = "sample"
        args.sliding_window = 10
        args.tot_num_per_mineral = 6
        args.noise_std = 3
        args.rand_shift = 0
        args.min_wave = 40
        args.max_wave = 1496
        if args.augmentation_mode == "none":
            args.augment_option = "none"
            args.tot_num_per_mineral = 0
        if args.augmentation_mode == "pure_noise":
            args.noise_std = 0.1
    if "organic_target" in args.raman_type:
        args.sliding_window = 10
        # args.tot_num_per_mineral = 21
        args.noise_std = 1
        args.min_wave = 0
        args.max_wave = 1024
        if args.raman_type == "organic_target":
            args.rand_shift = 3
            args.lower_bound = 1 + args.rand_shift * 3
            args.upper_bound = args.max_wave - args.min_wave - args.rand_shift * 3
        else:
            args.rand_shift = 0
            args.lower_bound = 0
            args.upper_bound = 1024
    if "bacteria" in args.raman_type:
        args.sliding_window = 0
        args.noise_std = 0
        args.max_wave = 1000
        args.min_wave = 0
        args.rand_shift = 0
        args.lower_bound, args.upper_bound = 0, 1000
    if args.augment_option == "none":
        args.tot_num_per_mineral = 0
        args.sliding_window = 0
        args.rand_shift = 0
        args.noise_std = 0
        model_tem = model_mom + "lr_%.5f" % args.learning_rate_init
    elif "sample" in args.augment_option:
        model_tem = model_mom + "lr_%.5f" % args.learning_rate_init
    model_tem = model_tem + "_Xception" + "_distance_aggregate_%s" % args.distance_aggregation
    model_tem += "_lrschedule_%s_l2regu_%.4f" % (args.lr_schedule, args.l2_regu_para)
    model_tem += "_stem_kernel_%d_ch_%d_channel_depth_%d_dropout_%s" % (args.stem_kernel,
                                                                        args.stem_max_dim,
                                                                        args.depth, args.dropout)
    if args.within_dropout:
        model_tem += "_drop_within_model"
    if not args.balanced or args.alpha != 0:
        model_tem += "_balance_pos_neg_%.3f" % args.alpha
    if args.check_ratio_on_datasplit:
        model_tem += "_balance_multiple"
    if args.separable_act:
        model_tem += "_real_separable_conv"
    if args.dataset == "RRUFF":
        model_tem += "_augment_single_spectrum_%s" % args.augmentation_mode
    model_dir = model_tem + "/"
    create_dir(model_dir)
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-------------------------------------------------------------------")
    if not args.random_leave_one_out:
        args.pre_define_tt_filenames = False
        train_objective = Train(args, model_dir, None, False, args.dir2load_data + "/")
        train_objective.run()
    else:
        if args.repeat_time == 0:
            dir2save_data = model_tem + "/data_splitting/"
            create_dir(dir2save_data)
            pre_define_tt_filenames = False
            model_dir_temp = model_tem + "/repeat_0/"
            create_dir(model_dir_temp)
            train_objective = Train(args, model_dir_temp, dir2save_data, pre_define_tt_filenames, args.dir2load_data+"/")
            train_objective.run()
        else:
            dir2load_data = model_tem + "/data_splitting/"
            model_dir_temp = model_tem + "/repeat_%d/" % args.repeat_time
            create_dir(model_dir_temp)
            pre_define_tt_filenames = True
            train_objective = Train(args, model_dir_temp, dir2load_data, pre_define_tt_filenames, args.dir2load_data+"/")
            train_objective.run()



