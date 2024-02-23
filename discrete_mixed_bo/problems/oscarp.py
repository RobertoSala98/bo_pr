#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Problems with only binary variables.
"""
from typing import Optional

import numpy as np
import pandas as pd
from torch import Tensor
from typing import Any, Dict, Optional
from numpy.random import randint
import torch

from discrete_mixed_bo.problems.base import DiscreteTestProblem
from botorch.test_functions.base import ConstrainedBaseTestProblem


class OscarP(DiscreteTestProblem, ConstrainedBaseTestProblem):
    dim = 5
    num_constraints = 2

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self._dataset = pd.read_csv('./discrete_mixed_bo/problems/data/oscarp.csv')
        self._keys = ["parallelism_ffmpeg-0","parallelism_librosa","parallelism_ffmpeg-1","parallelism_ffmpeg-2","parallelism_deepspeech"]
        self._bounds = [(2, 4), (2, 6), (2, 4), (2, 8), (2, 4)]
        self._target_column = "-cost"
        self.idxs_ = []
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=list(range(len(self._keys))),
        )

    def get_approximation(self, x_probe):

        if self._dataset is None:
            raise ValueError("dataset is empty in get_approximation()")

        min_distance = None
        approximations = []
        approximations_idxs = []

        dataset_np = self._dataset.values
        idx_cols = [self._dataset.columns.get_loc(c) for c in self._keys]
        for idx in range(dataset_np.shape[0]):
            row = dataset_np[idx, idx_cols]
            dist = np.linalg.norm(x_probe - row, 2)
            if min_distance is None or dist <= min_distance:
                if dist == min_distance:
                    approximations.append(row)
                    approximations_idxs.append(self._dataset.index[idx])
                else:
                    min_distance = dist
                    approximations = [row]
                    approximations_idxs = [self._dataset.index[idx]]

        ret_idx = randint(0, len(approximations_idxs))

        return approximations_idxs[ret_idx], approximations[ret_idx]


    def params_to_array(self, params):
        try:
            assert set(params) == set(self._keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self._keys)
            )
        return np.asarray([params[key] for key in self._keys])


    def find_point_in_dataset(self, params):

        params_ = {}
        for idx, key in enumerate(self._keys):
            params_[key] = params[idx]

        dataset_vals = self._dataset[self._keys].values
        x = self.params_to_array(params_)

        matches = np.where((dataset_vals == x).all(axis=1))[0]
        if len(matches) == 0:
            raise ValueError("{} not found in dataset".format(params_))
        idx = np.random.choice(matches)
        self.idxs_.append(idx)
        target_val = self._dataset.loc[idx, self._target_column]

        return target_val


    def evaluate_true(self, X: Tensor) -> Tensor:

        x_probe = X.numpy()

        y_ = []
        for elem in x_probe:
            idx_, x_ = self.get_approximation(elem)
            y_.append(self.find_point_in_dataset(x_))
            #print(elem, x_)

        return torch.tensor(y_, dtype=torch.float64)

    
    def evaluate_slack_true(self, X: Tensor) -> Tensor:

        import pdb; pdb.set_trace()

        return torch.stack([1.0, 1.0], dim=0)


