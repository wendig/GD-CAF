import numpy as np
import torch
from torch import nn
from tqdm import tqdm

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def compute_loss(model, test_dl, loss='mse', denormalize=True, only_first=False):
    model.to(dev)
    if loss.lower() == 'mae':
        loss_func = nn.functional.l1_loss
    else:
        loss_func = nn.functional.mse_loss
    factor = 1
    if denormalize:
        factor = 1 * 0.03651024401187897
    # go through test set

    with torch.no_grad():
        threshold = 0.0005  # rainfall: 0.0005 mm/h
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        loss_per_region = [0.0] * 16
        for x, y_true in tqdm(test_dl, position=0, leave=True):
            y_pred = model(x.to(dev))
            y_pred = y_pred.to(dev)
            y_true = y_true.to(dev)
            # Denormalize
            y_pred = y_pred.squeeze() * factor
            y_true = y_true.squeeze() * factor

            if only_first:
                if y_pred.dim() == 4:
                    y_pred = y_pred[:, 0, :, :]
                    y_true = y_true[:, 0, :, :]
                elif y_pred.dim() == 3:
                    y_pred = y_pred[0, :, :]
                    y_true = y_true[0, :, :]
            for i in range(len(loss_per_region)):
                tmp_pred, tmp_true = None, None
                if y_pred.dim() == 4:
                    tmp_pred = y_pred[:, i, :, :]
                    tmp_true = y_true[:, i, :, :]
                elif y_pred.dim() == 3:
                    tmp_pred = y_pred[i, :, :]
                    tmp_true = y_true[i, :, :]
                # Loss
                loss_per_region[i] += loss_func(tmp_pred, tmp_true, reduction="sum") / tmp_true.size(0)

            # loss
            loss_model += (loss_func(y_pred, y_true, reduction='sum') / y_true.size(0)).item()

            # convert to masks for comparison
            y_pred = y_pred > threshold
            y_true = y_true > threshold

            tn, fp, fn, tp = np.bincount(y_true.view(-1).cpu() * 2 + y_pred.view(-1).cpu(), minlength=4)

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
            hss = (total_tp * total_tn - total_fp * total_fn) / ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))
            del y_pred

        loss_model /= len(test_dl)
        loss_per_region = [(x / len(test_dl)).cpu().item() for x in loss_per_region]
    return loss_per_region, np.array(loss_model), precision, recall, accuracy, f1, csi, far, hss
