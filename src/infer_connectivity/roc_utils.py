import numpy as np
from sklearn.metrics import auc, average_precision_score, f1_score, roc_curve


def compute_roc_curve(true_conn, filters):
    """
    Compute roc curve and scores.

    Parameters
    ----------
    true_conn:
        Connectivity matrix, (input, output) or flattened (input,).
    filters:
        Response filters, (input, output, time) or (input, 1, time)

    Returns
    -------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    roc_auc : float
        Area under ROC curve
    ap : float
        Average precision score
    pred_conn : array
        Binary predictions using best F1 threshold
    best_f1 : float
        Best F1 score achieved
    """
    true_conn = true_conn.reshape(-1, 1)
    filters = filters.reshape(-1, filters.shape[-1])
    abs_argmax = np.argmax(np.abs(filters), axis=1)
    peak_filt = np.array(
        np.take_along_axis(filters, abs_argmax[:, np.newaxis], axis=1).squeeze(axis=1)
    )
    scores = np.abs(peak_filt)
    fpr, tpr, roc_thresh = roc_curve(true_conn, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(true_conn, scores)

    f1s = []
    for t in roc_thresh:
        preds = (scores >= t).astype(int)
        f1s.append(f1_score(true_conn, preds))

    best_f1_idx = np.argmax(f1s)
    best_f1 = f1s[best_f1_idx]
    best_t = roc_thresh[best_f1_idx]
    pred_conn = (scores >= best_t).astype(int)

    return fpr, tpr, roc_auc, ap, pred_conn, best_f1


def compute_filters(weights, basis):
    """Weights from GLM.coef_ or PopulationGLM.coef_"""
    kernels = basis.evaluate_on_grid(basis.window_size)[1]
    if weights.ndim == 1:
        # single neuron glm, [input, n_basis]
        weights = weights.reshape( -1, basis.n_basis_funcs)
        # [input, time]
        resp_filters = weights @ kernels.T
        # [input, 1, time]
        resp_filters = resp_filters[:, None]
    elif weights.ndim == 2:
        n_neurons = weights.shape[-1]
        # [input, n_basis, output]
        weights = weights.reshape(n_neurons, basis.n_basis_funcs, n_neurons)
        # [input, output, time]
        resp_filters = np.einsum("ikj,tk->ijt", weights, kernels)
    else:
        raise ValueError("Weights has wrong shape")
    return resp_filters