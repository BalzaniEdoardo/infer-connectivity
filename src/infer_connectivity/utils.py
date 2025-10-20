import pathlib
import re

import numpy as np

from . import load_model


def extract_coef_single_neu_glm(output_dir: str | pathlib.Path, pattern=None):
    output_dir = pathlib.Path(output_dir)
    if pattern is None:
        pattern = re.compile(r"best_.*neuron_(\d+).*\.npz")

    neu_ids = []
    for fh in output_dir.iterdir():
        found = re.search(pattern, fh.name)
        if found is not None:
            neu_ids.append(int(found.group(1)))
    neu_ids = sorted(neu_ids)
    reg_strengths = np.full(len(neu_ids), np.nan)
    coef = None
    for fh in output_dir.iterdir():
        found = re.search(pattern, fh.name)
        if found is not None:
            neu = int(found.group(1))
            model = load_model(fh)
            if coef is None:
                coef = np.zeros((*model.coef_.shape, len(neu_ids)))
            coef[..., neu_ids.index(neu)] = model.coef_
            reg_strengths[neu_ids.index(neu)] = model.regularizer_strength
    return np.array(neu_ids), coef, reg_strengths
