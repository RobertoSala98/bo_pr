#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import pandas as pd
from torch import Tensor
from typing import Any, Dict, Optional
from numpy.random import randint
import torch
import csv

from discrete_mixed_bo.problems.base import DiscreteTestProblem
from botorch.test_functions.base import ConstrainedBaseTestProblem


class Stereomatch(DiscreteTestProblem, ConstrainedBaseTestProblem):
    dim = 4
    num_constraints = 2

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        output_path: Optional[str] = ''
    ) -> None:
        self._dataset = pd.read_csv('./discrete_mixed_bo/problems/data/stereomatch.csv')
        self._keys = ["confidence", "hypo_step", "max_arm_length", "num_threads"]
        self._bounds = [(14, 64), (1, 3), (1, 16), (1, 32)]
        self._target_column = "-cost"
        self._bounds_column = "time"
        self.idxs_ = []
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=list(range(len(self._keys))),
        )

        self.output_path_csv = output_path.split(".pt")[0] + ".csv"
        header = ["index"] + self._keys + ["target", "feasible"]

        with open(self.output_path_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            

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

        with open(self.output_path_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for elem in x_probe:
                idx_, x_ = self.get_approximation(elem)
                y_.append(self.find_point_in_dataset(x_))
                #print(elem, x_)

                target_ = y_[-1]
                feasible_ = self._dataset.loc[idx_, self._bounds_column] >= 0 and self._dataset.loc[idx_, self._bounds_column] <= 17000

                csv_data = [idx_] + x_.tolist() + [target_] + [feasible_]
                writer.writerow(csv_data)

        return torch.tensor(y_, dtype=torch.float64)

    
    def evaluate_slack_true(self, X: Tensor) -> Tensor:

        bound_val = self._dataset.loc[self.idxs_, self._bounds_column].values
        g1 = torch.from_numpy(bound_val)
        g2 = torch.from_numpy(17000.0 - bound_val)

        self.idxs_ = []

        return torch.stack([g1, g2], dim=-1)