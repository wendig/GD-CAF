from argparse import Namespace
import torch
from metrics.metric import compute_loss
from metrics.persistence import print_persistent_metrics
from models.GD_CAF import GDCAF
from models.model_base import get_test_dataset
from models.unets import UNetDS_Attention


def evaluate(checkpoint_folder, checkpoint_name, denormalize=True, calc_persistence=True, only_first=False):
    test_losses = dict()
    # Load checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_folder + checkpoint_name)
    else:
        checkpoint = torch.load(checkpoint_folder + checkpoint_name, map_location=lambda storage, loc: storage)
    hparams = Namespace(**checkpoint["hyper_parameters"])
    # Model
    net = None
    if hparams.model == "GDCAF":
        net = GDCAF(hparams=hparams)
    elif hparams.model == "UNetDS_Attention":
        net = UNetDS_Attention(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")
    # Load Model
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    # Set path to test
    hparams.dataset_folder = "data/test"
    # Dataloader
    t_loader = get_test_dataset(
            dataset_path=hparams.dataset_folder,
            past_look=hparams.num_input_images,
            future_look=hparams.num_output_images,
            fast_dev_run=False,
            B=hparams.batch_size,
            cell_path=hparams.cell_path,
            cell_cutoff=hparams.cell_cutoff
    )
    # Persistence
    if calc_persistence:
        print_persistent_metrics(t_loader, only_first=only_first)
    # calculate loss
    lpr, loss_model, precision, recall, accuracy, f1, csi, far, hss = compute_loss(net, t_loader, 'mse', denormalize=denormalize, only_first=only_first)
    test_losses[checkpoint_name] = loss_model
    print()
    print(checkpoint_name)
    print(f"(MSE): {loss_model:.8f}, precision: {precision:.5f}, recall: {recall:.5f}, accuracy: {accuracy:.5f}, f1: {f1:.5f}, csi: {csi:.5f}, far: {far:.5f}, hss: {hss:.5f}")
    print(f"Loss per region: {lpr}")
    return test_losses


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    evaluate('db/saved_models/exp1/', 'GDCAF_epoch=56-val_loss=0.896607_L_2_K_4_kpl_2_in_6_out_6_cut_16_both_pooling.ckpt', denormalize=True, calc_persistence=False, only_first=False)
