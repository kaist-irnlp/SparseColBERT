# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import math

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def max_entropy(n, k):
    """The maximum entropy we could get with n units and k winners."""
    s = float(k) / n
    if 0.0 < s < 1.0:
        entropy = -s * math.log(s, 2) - (1 - s) * math.log(1 - s, 2)
    else:
        entropy = 0

    return n * entropy


def binary_entropy(x):
    """Calculate entropy for a list of binary random variables.

    :param x: (torch tensor) the probability of the variable to be 1.
    :return: entropy: (torch tensor) entropy, sum(entropy)
    """
    entropy = -x * x.log2() - (1 - x) * (1 - x).log2()
    entropy[x * (1 - x) == 0] = 0
    return entropy, entropy.sum()


def plot_duty_cycle(duty_cycle):
    """
    Create plot showing histogram of duty cycles

    :param duty_cycle: (torch tensor) the duty cycle of each unit
    :param filePath: (str) Full filename of image file
    """
    _, entropy = binary_entropy(duty_cycle)
    fig = plt.figure()
    bins = np.linspace(0.0, 0.3, 200)
    plt.hist(duty_cycle, bins, alpha=0.5, label="All cols")
    plt.title("Histogram of duty cycles, entropy=" + str(float(entropy)))
    plt.xlabel("Duty cycle")
    plt.ylabel("Number of units")
    return fig
