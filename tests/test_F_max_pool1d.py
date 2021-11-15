# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.max_pool1d(x, kernel_size=3)
        x = F.max_pool1d(x, kernel_size=4, stride=2, padding=2, dilation=1)
        x = F.max_pool1d(
            x, kernel_size=3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False
        )
        x = F.max_pool1d(
            x, kernel_size=5, stride=2, padding=2, dilation=1, return_indices=False, ceil_mode=True
        )
        x = F.max_pool1d(
            x, kernel_size=3, stride=1, padding=1, dilation=2, return_indices=False, ceil_mode=False
        )
        x = F.max_pool1d(
            x, kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=True
        )
        x, indices1 = F.max_pool1d(
            x, kernel_size=2, padding=1, dilation=1, return_indices=True, ceil_mode=False
        )
        x, indices2 = F.max_pool1d(
            x, kernel_size=5, stride=1, padding=2, dilation=1, return_indices=True, ceil_mode=True
        )
        return x, indices1, indices2


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128)

    a0, a1, a2 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_max_pool1d.pt")

    # torchscript to pnnx
    import os

    os.system("../src/pnnx test_F_max_pool1d.pt inputshape=[1,12,128]")

    # pnnx inference
    import test_F_max_pool1d_pnnx

    b0, b1, b2 = test_F_max_pool1d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
