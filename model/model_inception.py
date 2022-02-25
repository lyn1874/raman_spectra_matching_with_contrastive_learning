"""
Created on 11:18 at 09/07/2021
Clean the Xception code
@author: bo
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init


class Siamese(nn.Module):
    def __init__(self, wavenumber, aggregate, stem_kernel=21,
                 depth=128, stem_max_dim=64, dropout=False,
                 within_dropout=False, separable_act=False):
        super(Siamese, self).__init__()
        self.aggregate = aggregate
        self.wavenumber = wavenumber
        self.depth = depth
        self.dropout = dropout
        wave_dimension_dim = self.wavenumber // 2
        if self.aggregate != "wave_channel_dot_L1":
            wave_aggregate = wave_dimension_dim
        else:
            wave_aggregate = wave_dimension_dim * 2
        self.channel_aggregate = depth
        self.wave_aggregate = wave_aggregate
        self.extractor = Xception(self.wavenumber, stem_kernel, 2, act="leakyrelu",
                                  depth=depth, stem_max_dim=stem_max_dim,
                                  within_dropout=within_dropout, separable_act=separable_act)
        self.feature_dim = wavenumber // 2 * self.depth
        self.aggregate_channel_dot = nn.Conv1d(self.channel_aggregate, 1, 1)
        self.aggregate_channel_L1 = nn.Conv1d(self.channel_aggregate, 1, 1)
        self.aggregate_wave = nn.Linear(self.wave_aggregate, 1)

    def forward(self, x):
        features = self.extractor(x)
        num_pair_sample = len(features)
        feature_original = features[:num_pair_sample // 2]
        feature_paired = features[num_pair_sample // 2:]
        feature_diff_l1 = (feature_original - feature_paired).abs()
        feature_diff_dot = feature_original * feature_paired
        if self.dropout:
            feature_diff_l1 = nn.Dropout(inplace=True)(feature_diff_l1)
            feature_diff_dot = nn.Dropout(inplace=True)(feature_diff_dot)
        distance_wave_dot = self.aggregate_channel_dot(feature_diff_dot).squeeze(1)
        distance_wave_l1 = self.aggregate_channel_L1(feature_diff_l1).squeeze(1)
        distance = self.aggregate_wave(torch.cat([distance_wave_l1, distance_wave_dot], dim=1))
        return distance

    def forward_test(self, x, unuse):
        batch_size = 80
        num_sample = int(np.ceil(len(x) / batch_size))
        features = []
        for i in range(num_sample):
            if i != num_sample - 1:
                _features = self.extractor(x[i * batch_size: (i + 1) * batch_size])
            else:
                _features = self.extractor(x[i * batch_size:])
            features.append(_features)
        features = torch.cat(features, dim=0)
        return features

    def _dropout(self, features, num_dropout_samples=0):
        """Apply dropout during the test time
        Args:
            features: [batch_size, channel_size, wavenumber size]
            num_dropout_samples: int, decides the number of dropout times
        Output:
            features: concatenate along the batch size dimension
        """
        if num_dropout_samples == 0:
            return features
        else:
            return nn.Dropout(inplace=True)(features)

    def forward_test_batch(self, x, features, num_dropout_sample=0):
        """Give the predicted value given a batch of input,
        This is mainly for reducing inference time
        """
        if x.shape[1] == 1:
            features_test = self.extractor(x)
        else:
            features_test = x
        distance = []
        for v in features_test:
            _feature_l1 = (v - features).abs()
            _feature_dot = v * features
            _distance_l1 = self.aggregate_channel_L1(_feature_l1).squeeze(1)
            _distance_dot = self.aggregate_channel_dot(_feature_dot).squeeze(1)
            _distance = self.aggregate_wave(torch.cat([_distance_l1, _distance_dot], dim=1))
            distance.append(_distance.permute(1, 0))  # [batch_size, num_reference_spectra]
        distance_prediction = torch.cat(distance)
        return distance_prediction

    def forward_extract_feature_per_level(self, x):
        x_init, x_pool, x, x_block1, x_block2 = self.extractor.forward_multilevel_feature(x)
        return x_init, x_pool, x, x_block1, x_block2

    def forward_multilevel_similarity(self, x, reference_features):
        dot_similarity, l1similarity = self.extractor.forward_similarity_on_multilevel_features(x, reference_features)
        return dot_similarity, l1similarity


def calc_num_param(model):
    num_param = 0.0
    for name, p in model.named_parameters():
        _num = np.prod(p.shape)
        num_param += _num
    print("There are %.2f million parameters" % (num_param / 1e+6))


class SeparableConv1dACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        """Separable Convolutional Network
        If the input shape is : [2, 32, 128], and we want to get output size of [2, 64, 128] with kernel 3.
        In the normal convolutional operation, the number of parameters is:
            32 * 64 * 3
        In the separable convolution, the number of parameter is:
            1 * 1 * 3 * 32 + 1 * 32 * 64 = 3 * 32 * (1 + 64/3) round to 3 * 32 * 21, which has 3 times less number of
            parameters compared to the original operation
        """
        super(SeparableConv1dACT, self).__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)
        self.conv1.apply(normal_init)
        self.pointwise.apply(normal_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionStemBlock(nn.Module):
    def __init__(self, kernel, depth=1, max_dim=64):
        super(XceptionStemBlock, self).__init__()
        if max_dim == 64 or max_dim == 128:
            input_dim = [1, 32]
            output_dim = [32, 64]
        elif max_dim == 32:
            input_dim = [1, 16]
            output_dim = [16, 32]

        act = nn.LeakyReLU(0.3)
        self.depth = depth
        self.stem_1 = nn.Sequential()
        input_channel = input_dim[0]
        output_channel = output_dim[0]
        pad = int((kernel - 1) // 2)
        for i in range(2):
            self.stem_1.add_module("stem1_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                        output_channel,
                                                                        kernel_size=kernel,
                                                                        stride=1,
                                                                        padding=pad))
            self.stem_1.add_module("stem1_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
            self.stem_1.add_module("stem1_act_%d" % (i + 1), act)
            input_channel = output_channel

        output_channel = output_dim[1]
        if depth == 2:
            self.stem_2 = nn.Sequential()
            for i in range(2):
                self.stem_2.add_module("stem2_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                            output_channel,
                                                                            kernel_size=kernel,
                                                                            stride=1,
                                                                            padding=pad))
                self.stem_2.add_module("stem2_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
                self.stem_2.add_module("stem2_act_%d" % (i + 1), act)
                input_channel = output_channel
            self.stem_2.apply(normal_init)

        self.stem_1.apply(normal_init)

    def forward(self, x):
        x = self.stem_1(x)
        x = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x = self.stem_2(x)
        return x

    def forward_test(self, x):
        x = self.stem_1(x)
        x_pool = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x_further = self.stem_2(x_pool)
        else:
            x_further = x_pool
        return x, x_pool, x_further


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, kernel_size,
                 stride=1, act="relu", start_with_act=True, grow_first=True,
                 separable_act=False):
        super(XceptionBlock, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.skipbn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.3, inplace=True)
        else:
            print("------The required activation function doesn't exist--------")
        self.separable_act = separable_act
        rep = []
        filters = in_channels
        if grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(in_channels, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        for i in range(repeats)[1:]:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        if not grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))

        if not start_with_act:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        return x + skip


class Xception(nn.Module):
    def __init__(self, wavenumber, stem_kernel, num_xception_block=2, act="leakyrelu", depth=128,
                 stem_max_dim=64, within_dropout=False, separable_act=False):
        super(Xception, self).__init__()
        self.depth = depth
        self.num_xception_block = num_xception_block
        self.stem = XceptionStemBlock(stem_kernel, 2, stem_max_dim)
        self.block1 = XceptionBlock(stem_max_dim, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=False, grow_first=True,
                                    separable_act=separable_act)
        self.block2 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=True, grow_first=True,
                                    separable_act=separable_act)
        if num_xception_block == 3:
            self.block3 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                        stride=1, act=act, start_with_act=True, grow_first=True,
                                        separable_act=separable_act)
        if num_xception_block == 2:
            self.feature_dimension = wavenumber // 2
        elif num_xception_block == 3:
            self.feature_dimension = wavenumber // 4
        self.within_dropout = within_dropout

    def forward(self, x):
        x = self.stem(x)
        if self.within_dropout:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block1(x)
        if self.num_xception_block == 3:
            x = nn.MaxPool1d(2)(x)
        if self.within_dropout:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block2(x)
        if self.num_xception_block == 3:
            x = self.block3(x)
        return x

    def test_dropout(self, x, num_sample):
        """Get the MC Dropout probability of the prediction at the test time
        Args:
            x: feature maps from the last block in the inception network
            num_sample: the generated number of samples
        """
        x_g = []
        for i in range(num_sample):
            _x = nn.Dropout(p=0.5)(x)
            x_g.append(_x)
        x_g = torch.cat(x_g, dim=0).reshape([num_sample, self.depth, self.feature_dimension])
        return x_g

    def forward_multilevel_feature(self, x):
        x_init, x_pool, x = self.stem.forward_test(x)
        x_block1 = self.block1(x)
        x_block2 = self.block2(x_block1)
        return x_init, x_pool, x, x_block1, x_block2

    def forward_similarity_on_multilevel_features(self, test_data, reference_features):
        """Calculate the simialrity on multiple levels of features
        Maybe I can add auxiliary tasks along the network, because this is similar to the auxiliary task as well?
        Args:
            test_data: [num_test_data, 1, wave_number], tensor
            reference_features: dictionary, where the keys are "level-%d" % i for i in [1, 2, 3, 4]
        """
        _, x_pool, x, x_block1, x_block2 = self.forward_multilevel_feature(test_data)  # [100, channel, wavenumber]
        test_features = [x_pool, x, x_block1, x_block2]
        dot_similarity = {}
        l1norm_similarity = {}
        for key in reference_features.keys():
            dot_similarity[key] = []
            l1norm_similarity[key] = []
        for i, single_key in enumerate(reference_features.keys()):
            _reference_feature = reference_features[single_key]
            _test_feature = test_features[i]
            for j, _s_t_feat in enumerate(_test_feature):
                _dot_product = (_s_t_feat * _reference_feature)  # [num_reference_data, channel, wavenumber]
                _l1norm_value = (_s_t_feat - _reference_feature).abs()  # [num_reference_data, channel, wavenumber]
                dot_similarity[single_key].append(_dot_product.sum(dim=(-1, -2)))
                l1norm_similarity[single_key].append(_l1norm_value.sum(dim=(-1, -2)))
        return dot_similarity, l1norm_similarity


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        init.normal_(m.weight, 0, 0.05)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
