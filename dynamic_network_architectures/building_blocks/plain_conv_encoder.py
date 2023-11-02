import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
    StackedConvBlocksSeparateNorm,
    MultiInputModule,
    STN_block,
)
from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
)
from dynamic_network_architectures.building_blocks.CMX_utils import (
    FeatureRectifyModule as FRM,
)
from dynamic_network_architectures.building_blocks.CMX_utils import (
    FeatureFusionModuleStage1 as FFMs1,
)
from dynamic_network_architectures.building_blocks.CMX_utils import (
    Tmp,
    Tmp2,
    Tmp2_3modals,
    EfficientCrossAttentionHeadV2,
    PyTorchMultiHeadCrossAttention,
)


class PlainConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvEncoderSeparateNorm(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        num_separate_norms = input_channels
        input_channels = 1
        self.num_separate_norms = num_separate_norms
        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocksSeparateNorm(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                    num_separate_norms,
                )
            )
            stages.append(MultiInputModule(stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x, norm_idx):
        ret = []
        for s in self.stages:
            x = s(x, norm_idx)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], MultiInputModule):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvEncoderSeparateNormV2(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        num_separate_norms = input_channels
        input_channels = 1
        self.num_separate_norms = num_separate_norms
        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocksSeparateNorm(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                    num_separate_norms,
                )
            )
            stages.append(MultiInputModule(stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=1)
        ret = []
        for s in self.stages:
            x0 = s(x0, 0)
            x1 = s(x1, 1)
            ret.append(x0 + x1)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], MultiInputModule):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderCMX(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
        reduction: int = 1,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages = []
        FRMs = []
        for s in range(n_stages):
            stage_modules = []

            # NOTE: hardcoded for two modalities
            for _ in range(2):
                modal_modules = []
                if pool == "max" or pool == "avg":
                    if (
                        (isinstance(strides[s], int) and strides[s] != 1)
                        or isinstance(strides[s], (tuple, list))
                        and any([i != 1 for i in strides[s]])
                    ):
                        modal_modules.append(
                            get_matching_pool_op(conv_op, pool_type=pool)(
                                kernel_size=strides[s], stride=strides[s]
                            )
                        )
                    conv_stride = 1
                elif pool == "conv":
                    conv_stride = strides[s]
                else:
                    raise RuntimeError()
                modal_modules.append(
                    StackedConvBlocks(
                        n_conv_per_stage[s],
                        conv_op,
                        input_channels,
                        features_per_stage[s],
                        kernel_sizes[s],
                        conv_stride,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                        nonlin_first,
                    )
                )
                stage_modules.append(nn.ModuleList(modal_modules))
            stages.append(nn.ModuleList(stage_modules))
            FRMs.append(FRM(features_per_stage[s], reduction=reduction))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.FRMs = nn.ModuleList(FRMs)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for s, frm in zip(self.stages, self.FRMs):
            x0 = s[0][0](x0)
            x1 = s[1][0](x1)
            x0, x1 = frm(x0, x1)
            ret.append([x0, x1])
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderFRMFFMs1(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
        reduction: int = 1,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages = []
        FRMs = []
        FFMs_stage1 = []
        for s in range(n_stages):
            stage_modules = []

            # NOTE: hardcoded for two modalities
            for _ in range(2):
                modal_modules = []
                if pool == "max" or pool == "avg":
                    if (
                        (isinstance(strides[s], int) and strides[s] != 1)
                        or isinstance(strides[s], (tuple, list))
                        and any([i != 1 for i in strides[s]])
                    ):
                        modal_modules.append(
                            get_matching_pool_op(conv_op, pool_type=pool)(
                                kernel_size=strides[s], stride=strides[s]
                            )
                        )
                    conv_stride = 1
                elif pool == "conv":
                    conv_stride = strides[s]
                else:
                    raise RuntimeError()
                modal_modules.append(
                    StackedConvBlocks(
                        n_conv_per_stage[s],
                        conv_op,
                        input_channels,
                        features_per_stage[s],
                        kernel_sizes[s],
                        conv_stride,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                        nonlin_first,
                    )
                )
                stage_modules.append(nn.ModuleList(modal_modules))
            stages.append(nn.ModuleList(stage_modules))
            FRMs.append(FRM(features_per_stage[s], reduction=reduction))
            FFMs_stage1.append(
                Tmp2(n_embed=features_per_stage[s], num_heads=2 ** (1 + s))
            )
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.FRMs = nn.ModuleList(FRMs)
        self.FFMs_stage1 = nn.ModuleList(FFMs_stage1)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for s, frm, ffms1 in zip(self.stages, self.FRMs, self.FFMs_stage1):
            x0 = s[0][0](x0)
            x1 = s[1][0](x1)
            x0, x1 = frm(x0, x1)
            x0, x1 = ffms1(x0, x1)
            ret.append([x0, x1])
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderFRMFFMs1_softmaxExp(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
        reduction: int = 1,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages = []
        FRMs = []
        FFMs_stage1 = []
        for s in range(n_stages):
            stage_modules = []

            # NOTE: hardcoded for two modalities
            for _ in range(2):
                modal_modules = []
                if pool == "max" or pool == "avg":
                    if (
                        (isinstance(strides[s], int) and strides[s] != 1)
                        or isinstance(strides[s], (tuple, list))
                        and any([i != 1 for i in strides[s]])
                    ):
                        modal_modules.append(
                            get_matching_pool_op(conv_op, pool_type=pool)(
                                kernel_size=strides[s], stride=strides[s]
                            )
                        )
                    conv_stride = 1
                elif pool == "conv":
                    conv_stride = strides[s]
                else:
                    raise RuntimeError()
                modal_modules.append(
                    StackedConvBlocks(
                        n_conv_per_stage[s],
                        conv_op,
                        input_channels,
                        features_per_stage[s],
                        kernel_sizes[s],
                        conv_stride,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                        nonlin_first,
                    )
                )
                stage_modules.append(nn.ModuleList(modal_modules))
            stages.append(nn.ModuleList(stage_modules))
            FRMs.append(FRM(features_per_stage[s], reduction=reduction))
            FFMs_stage1.append(
                Tmp2(
                    n_embed=features_per_stage[s],
                    num_heads=2 ** (1 + s),
                    attention_block=EfficientCrossAttentionHeadV2,
                )
            )
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.FRMs = nn.ModuleList(FRMs)
        self.FFMs_stage1 = nn.ModuleList(FFMs_stage1)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for s, frm, ffms1 in zip(self.stages, self.FRMs, self.FFMs_stage1):
            x0 = s[0][0](x0)
            x1 = s[1][0](x1)
            x0, x1 = frm(x0, x1)
            x0, x1 = ffms1(x0, x1)
            ret.append([x0, x1])
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderCrossAttn(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages = []
        cross_attn = []
        for s in range(n_stages):
            stage_modules = []

            # NOTE: hardcoded for two modalities
            for _ in range(2):
                modal_modules = []
                if pool == "max" or pool == "avg":
                    if (
                        (isinstance(strides[s], int) and strides[s] != 1)
                        or isinstance(strides[s], (tuple, list))
                        and any([i != 1 for i in strides[s]])
                    ):
                        modal_modules.append(
                            get_matching_pool_op(conv_op, pool_type=pool)(
                                kernel_size=strides[s], stride=strides[s]
                            )
                        )
                    conv_stride = 1
                elif pool == "conv":
                    conv_stride = strides[s]
                else:
                    raise RuntimeError()
                modal_modules.append(
                    StackedConvBlocks(
                        n_conv_per_stage[s],
                        conv_op,
                        input_channels,
                        features_per_stage[s],
                        kernel_sizes[s],
                        conv_stride,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                        nonlin_first,
                    )
                )
                stage_modules.append(nn.ModuleList(modal_modules))
            stages.append(nn.ModuleList(stage_modules))
            cross_attn.append(Tmp2(n_embed=features_per_stage[s], num_heads=8))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.cross_attn = nn.ModuleList(cross_attn)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for s, cross_attn in zip(self.stages, self.cross_attn):
            x0 = s[0][0](x0)
            x1 = s[1][0](x1)
            x0, x1 = cross_attn(x0, x1)
            ret.append(x0 + x1)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderCrossAttn3modals(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages = []
        cross_attn = []
        for s in range(n_stages):
            stage_modules = []

            # NOTE: hardcoded for three modalities
            for _ in range(3):
                modal_modules = []
                if pool == "max" or pool == "avg":
                    if (
                        (isinstance(strides[s], int) and strides[s] != 1)
                        or isinstance(strides[s], (tuple, list))
                        and any([i != 1 for i in strides[s]])
                    ):
                        modal_modules.append(
                            get_matching_pool_op(conv_op, pool_type=pool)(
                                kernel_size=strides[s], stride=strides[s]
                            )
                        )
                    conv_stride = 1
                elif pool == "conv":
                    conv_stride = strides[s]
                else:
                    raise RuntimeError()
                modal_modules.append(
                    StackedConvBlocks(
                        n_conv_per_stage[s],
                        conv_op,
                        input_channels,
                        features_per_stage[s],
                        kernel_sizes[s],
                        conv_stride,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                        nonlin_first,
                    )
                )
                stage_modules.append(nn.ModuleList(modal_modules))
            stages.append(nn.ModuleList(stage_modules))
            cross_attn.append(Tmp2_3modals(n_embed=features_per_stage[s], num_heads=8))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.cross_attn = nn.ModuleList(cross_attn)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1, x2):
        ret = []
        for s, cross_attn in zip(self.stages, self.cross_attn):
            x0 = s[0][0](x0)
            x1 = s[1][0](x1)
            x2 = s[2][0](x2)
            x0, x1, x2 = cross_attn(x0, x1, x2)
            ret.append(x0 + x1 + x2)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvEncoderSeparateNormCMX(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        num_separate_norms = input_channels
        input_channels = 1
        self.num_separate_norms = num_separate_norms
        stages = []
        FRMs = []
        for s in range(n_stages):
            stage_modules = []
            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocksSeparateNorm(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                    num_separate_norms,
                )
            )
            stages.append(MultiInputModule(stage_modules))
            FRMs.append(FRM(features_per_stage[s]))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.FRMs = nn.ModuleList(FRMs)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for s, frm in zip(self.stages, self.FRMs):
            x0 = s(x0, 0)
            x1 = s(x1, 1)
            x0, x1 = frm(x0, x1)
            ret.append([x0, x1])
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], MultiInputModule):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class STN_fusion_blocks(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        num_separate_norms = input_channels
        input_channels = 1
        self.num_separate_norms = num_separate_norms
        STN_fusion_blocks = []
        for s in range(n_stages):
            STN_fusion_blocks.append(
                STN_block(features_per_stage[s], kernel_sizes[s:], strides[s:])
            )

        self.STN_fusion_blocks = STN_fusion_blocks


class PlainConvSeparateEncoderPyTorchNativeAttention(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        input_channels = 1
        stages_modal0 = []
        stages_modal1 = []
        cross_attn = []
        patch_dims = [
            [2, 8, 8],
            [2, 4, 4],
            [2, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        shapes = [
            [40, 192, 224],
            [40, 96, 112],
            [40, 48, 56],
            [20, 24, 28],
            [10, 12, 14],
            [5, 6, 7],
        ]
        for s in range(n_stages):
            stage_modules_modal0 = []
            stage_modules_modal1 = []

            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules_modal0.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                    stage_modules_modal1.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules_modal0.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stage_modules_modal1.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stages_modal0.append(nn.Sequential(*stage_modules_modal0))
            stages_modal1.append(nn.Sequential(*stage_modules_modal1))
            if s < 3:
                cross_attn.append(None)
            else:
                cross_attn.append(
                    PyTorchMultiHeadCrossAttention(
                        in_channels=features_per_stage[s],
                        num_heads=8,
                        patch_depth=patch_dims[s][0],
                        patch_height=patch_dims[s][1],
                        patch_width=patch_dims[s][2],
                        shp=shapes[s],
                        kernel_size=None,
                        stride=None,
                    )
                )
            input_channels = features_per_stage[s]

        plot = False
        if plot:
            device = torch.device("cuda:0")
            import hiddenlayer as hl
            import os

            att = PyTorchMultiHeadCrossAttention(
                in_channels=features_per_stage[0],
                num_heads=8,
                patch_depth=patch_dims[0][0],
                patch_height=patch_dims[0][1],
                patch_width=patch_dims[0][2],
                shp=shapes[0],
                kernel_size=None,
                stride=None,
            ).to(device)

            output_folder = r"/media/medical/gasperp/projects/nnUnetv2_clone/dynamic-network-architectures/dynamic_network_architectures"
            g = hl.build_graph(
                att,
                (
                    torch.rand((1, 32, 40, 192, 224), device=device),
                    torch.rand((1, 32, 40, 192, 224), device=device),
                ),
                transforms=None,
            )
            g.save(os.path.join(output_folder, "PyTorchMultiHeadCrossAttention.pdf"))

        # TODO make this modal_modules more transparent
        # and validate the architecture by checking the architecture plot
        # this is important!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.stages_modal0 = nn.ModuleList(stages_modal0)
        self.stages_modal1 = nn.ModuleList(stages_modal1)
        self.cross_attn = nn.ModuleList(cross_attn)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        for i in cross_attn:
            if i is not None:
                print(sum([np.prod(p.size()) for p in i.parameters()]))

    def forward(self, x0, x1):
        ret = []
        for sm0, sm1, cross_attn in zip(
            self.stages_modal0, self.stages_modal1, self.cross_attn
        ):
            x0 = sm0(x0)
            x1 = sm1(x1)
            if cross_attn is not None:
                x0, x1 = cross_attn(x0, x1)
            ret.append(x0 + x1)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvSeparateEncoderCrossAttnV4(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_conv_per_stage) == n_stages
        ), "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, (
            "strides must have as many entries as we have resolution stages (n_stages). "
            "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        )

        cross_attn = []
        stages_modal0 = []
        stages_modal1 = []

        for s in range(n_stages):
            stage_modules_modal0 = []
            stage_modules_modal1 = []

            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or isinstance(strides[s], (tuple, list))
                    and any([i != 1 for i in strides[s]])
                ):
                    stage_modules_modal0.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                    stage_modules_modal1.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules_modal0.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stage_modules_modal1.append(
                StackedConvBlocks(
                    n_conv_per_stage[s],
                    conv_op,
                    input_channels,
                    features_per_stage[s],
                    kernel_sizes[s],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            stages_modal0.append(nn.Sequential(*stage_modules_modal0))
            stages_modal1.append(nn.Sequential(*stage_modules_modal1))
            cross_attn.append(
                Tmp2(
                    n_embed=features_per_stage[s],
                    num_heads=8,
                    # attention_block=EfficientCrossAttentionHeadV2,
                )
            )
            input_channels = features_per_stage[s]

        self.stages_modal0 = nn.ModuleList(stages_modal0)
        self.stages_modal1 = nn.ModuleList(stages_modal1)
        self.cross_attn = nn.ModuleList(cross_attn)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x0, x1):
        ret = []
        for sm0, sm1, cross_attn in zip(
            self.stages_modal0, self.stages_modal1, self.cross_attn
        ):
            x0 = sm0(x0)
            x1 = sm1(x1)
            x0, x1 = cross_attn(x0, x1)
            ret.append(x0 + x1)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
