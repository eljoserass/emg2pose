# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import pytest
import torch
from emg2pose.kinematics import TorchNamedTuple, forward_kinematics


@pytest.mark.parametrize("requires_grad", [True, False])
def test_torch_named_tuple(requires_grad):
    class TensorTuple(NamedTuple):
        value: torch.Tensor

    tensors = TensorTuple(torch.tensor([1.0, 2.0]))
    module = TorchNamedTuple(tensors, requires_grad=requires_grad)

    assert module._fields == tensors._fields
    torch.testing.assert_close(module.value, tensors.value)
    assert module.value.requires_grad is requires_grad
    assert list(module.parameters()) == [module.value]

    module.to(dtype=torch.float64)
    assert module.value.dtype == torch.float64


def test_basic_default_forward_kinematics():
    joint_angles = torch.randn((2, 20, 4))  # batch, 20 joints, 4 time steps
    lp = forward_kinematics(joint_angles)
    assert lp.shape == (2, 4, 21, 3)  # batch, 4 time steps, 21 joints, 3d
