import pathlib
import pickle

import jax
import matplotlib.pyplot as plt
import nemos as nmo
import networkx as nx
import numpy as np
import pynapple as nap
from numba import config


def compute_perievent(rate: nap.TsdFrame, spike_times: nap.Ts, window_size_bin):
    idxs = np.searchsorted(rate.t, spike_times.t)
    in_window = (idxs > window_size_bin) & (idxs < len(rate) - window_size_bin)
    out_left = idxs <= window_size_bin
    out_right = idxs >= len(rate) - window_size_bin
    out = np.full((len(idxs), 2 * window_size_bin, *rate.d.shape[1:]), np.nan)
    slices = np.c_[idxs[in_window] - window_size_bin, idxs[in_window] + window_size_bin]
    out[in_window] = np.stack([rate.d[start:end] for start, end in slices])

    only_left = out_left & (~out_right)
    only_right = out_right & (~out_left)
    both_sides = out_left & out_right
    if np.any(only_left):
        deltas = window_size_bin - idxs[only_left]
        out_idx = np.where(only_left)[0]
        for k, d in enumerate(deltas):
            out[out_idx[k]][d:] = rate.d[: idxs[only_left][k] + window_size_bin]
    elif np.any(only_right):
        deltas = window_size_bin - (len(rate) - idxs[only_right])
        out_idx = np.where(only_right)[0]
        for k, d in enumerate(deltas):
            out[out_idx[k]][:d] = rate.d[idxs[only_right][k] - window_size_bin :]
    if np.any(both_sides):
        out_idx = np.where(both_sides)[0]
        for k in range(len(out_idx)):
            idx = idxs[both_sides][k]
            available_left = idx
            available_right = len(rate) - idx

            # Calculate how much we can take from each side
            left_take = min(window_size_bin, available_left)
            right_take = min(window_size_bin, available_right)

            # Place data in center of window
            center_start = window_size_bin - left_take
            center_end = window_size_bin + right_take

            out[out_idx[k]][center_start:center_end] = rate.d[
                idx - left_take : idx + right_take
            ]

    return out


jax.config.update("jax_enable_x64", True)

path_simulations = (
    pathlib.Path(__file__).parent.parent / "graph0_simulation_spikes.pckl"
)
path_graph = pathlib.Path(__file__).parent.parent / "graph0.graphml"

G = nx.read_graphml(path_graph)
connected = [
    (min(int(u), int(v)), max(int(u), int(v))) for u, v, data in G.edges(data=True)
]

sim_time = 40  # 400, 1000...
n_neurons = 100
binsize = 0.0003
history_window = 0.014
window_size = int(history_window / binsize)


n_basis_funcs = 4
c = 1.3

spikes = pickle.load(open(path_simulations, "rb"))

spikes_tsgroup = nap.TsGroup(
    {n: nap.Ts(np.array(spikes[n]) / 1000) for n in range(len(spikes))}
)

# inspect ccgs
crosscorrs = nap.compute_crosscorrelogram(
    group=spikes_tsgroup, binsize=0.3, windowsize=14, time_units="ms"  # ms
)

# for fignum in range(10):
#     f, axs = plt.subplots(5, 4, figsize=(10, 8))
#     for i, k in enumerate(connected[20*fignum:20*(fignum+1)]):
#         row = i // 4
#         col = i % 4
#         ax = axs[row, col]
#         ax.plot(crosscorrs[k])
#     plt.show()


counts = spikes_tsgroup.count(binsize)
basis = nmo.basis.RaisedCosineLogConv(4, window_size)
X = basis.compute_features(counts)
model = nmo.glm.GLM(
    observation_model="Bernoulli",
    regularizer="Lasso",
    regularizer_strength=10**-6,
    solver_kwargs={"tol": 10**-12},
)
model.fit(X, counts[:, 9])

rate = model.predict(X) / binsize

# config.DISABLE_JIT=True
aligned_rate = compute_perievent(rate, spikes_tsgroup[0], window_size)
aligned_counts = compute_perievent(counts[:, 9], spikes_tsgroup[0], window_size)
f, axs = plt.subplots(1, 1)
axs.plot(aligned_rate.mean(axis=0))
axs.plot(aligned_counts.mean(axis=0) / binsize)
plt.show()
# plt.plot(basis.evaluate_on_grid(window_size)[1].dot(model.coef_[:4]))
# plt.show()

nap.compute_perievent_continuous(
    rate, spikes_tsgroup[9], minmax=(-14 * 0.001, 14 * 0.001), time_units="ms"
)
