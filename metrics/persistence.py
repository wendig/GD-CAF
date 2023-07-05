import numpy as np
from torch import nn
from tqdm import tqdm


def persistence_prediction(a):
    """
        Predicts n sequence into the future, but they are the same as the last training data
    :param a: training data
    :return: prediction
    """
    return a[:, -1, :, :, :]


def get_persistence_metrics(test_dl, only_first=False):
    loss_func = nn.functional.mse_loss

    factor = 1 * 0.03651024401187897  # == 1/max(dataset)
    threshold = 0.0005  # 0.0005
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model, precision, recall, accuracy, f1, csi, far, hss = [0.0] * 8
    loss_per_region = [0.0] * 16

    for x, y_true in tqdm(test_dl, position=0, leave=True):
        y_pred = persistence_prediction(x)
        # denormalize
        y_pred_adj = y_pred.squeeze() * factor
        y_true_adj = y_true.squeeze() * factor

        if only_first:
            if y_pred.squeeze().dim() == 4:
                y_pred_adj = y_pred_adj[:, 0, :, :]
                y_true_adj = y_true_adj[:, 0, :, :]
            elif y_pred.squeeze().dim() == 3:
                y_pred_adj = y_pred_adj[0, :, :]
                y_true_adj = y_true_adj[0, :, :]

        #for i in range(len(loss_per_region)):
        #    tmp_pred, tmp_true = None, None
        #    if y_pred_adj.dim() == 4:
        #        tmp_pred = y_pred_adj[:, i, :, :]
        #        tmp_true = y_true_adj[:, i, :, :]
        #    elif y_pred_adj.dim() == 3:
        #        tmp_pred = y_pred_adj[i, :, :]
        #        tmp_true = y_true_adj[i, :, :]
        #    # Loss
        #    loss_per_region[i] += loss_func(tmp_pred, tmp_true, reduction="sum") / tmp_true.size(0)
        # Loss
        loss_model += loss_func(y_pred_adj, y_true_adj, reduction="sum") / y_true.size(0)
        # convert to masks for comparison
        y_pred_mask = y_pred_adj > threshold
        y_true_mask = y_true_adj > threshold

        tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        # get metrics for sample
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = (total_tp * total_tn - total_fp * total_fn) / (
                (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))

    loss_model /= len(test_dl)
    loss_per_region = [x / len(test_dl) for x in loss_per_region]
    return loss_per_region, loss_model, precision, recall, accuracy, f1, csi, far, hss


def print_persistent_metrics(t_loader, only_first=False):
    lpr, loss_model, precision, recall, accuracy, f1, csi, far, hss = get_persistence_metrics(t_loader, only_first)
    print()
    print(f"Loss Persistence (MSE): {loss_model:.8f}, precision: {precision:.5f}, recall: {recall:.5f}, accuracy: {accuracy:.5f}, f1: {f1:.5f}, csi: {csi:.5f}, far: {far:.5f}, hss: {hss:.5f}")
