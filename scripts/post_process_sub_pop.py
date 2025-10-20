import os
import pathlib
import json
import warnings

import nemos as nmo
import re
import pandas as pd
import seaborn as sns

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import ListedColormap, to_rgb
from nemos.regularizer import GroupLasso

from infer_connectivity import load_model
from infer_connectivity.roc_utils import compute_filters, compute_roc_curve
from pathlib import Path

GRADED_COLOR_LIST = [
    "navy",
    "blue",
    "royalblue",
    "cornflowerblue",
    "skyblue",
    "lightblue",
    "aquamarine",
    "mediumseagreen",
    "limegreen",
    "yellowgreen",
    "gold",
    "orange",
    "darkorange",
    "tomato",
    "orangered",
    "red",
    "crimson",
    "deeppink",
    "magenta",
]


base_dir = Path("/Users/ebalzani/Code/infer-connectivity/infer-connectivity/")
graph_file = base_dir / "simulations/sonica-sept-25-2025/graph0-sonica-sept-26-2026.graphml"
fit_id = "sonica-oct-8-2025-400-seconds"
output_directory = base_dir / "scripts" / "output" / fit_id
config_directory = base_dir / "scripts" / "configs" / fit_id
simulation_directory = base_dir / "simulations" / fit_id


# get conf
conf = None
for conf in config_directory.iterdir():
    if conf.suffix == ".json":
        break
