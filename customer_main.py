import math
import time
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, LambdaLR

from GDN_models.GDN import GDN
from customer_model import MTAD_Transform
from eval_methods import *
from AE.ae_model import ED

from args import get_parser
from utils import *
from my_model import MTAD_MODEL, MTAD_MODEL2, MTAD_MODEL3, MTAD_MODEL4
from TranAD.models import TranAD
from VAE.vae_model import VAE_MODEL
import torch.nn.functional as F

import matplotlib.pyplot as plt


def vae_loss(x, mu_pos, log_var_pos, x_pos_rec, x_disc_pos, x_disc_neg, mu_neg=None, log_var_neg=None):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    MSE = torch.mean(F.mse_loss(x_pos_rec, x, reduction='none'), dim=(1, 2))

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = torch.mean(kl_divergence(mu_pos, log_var_pos, torch.tensor(0), torch.tensor(1)), dim=1)
    # KLD = 0.5 * torch.sum(torch.exp(log_var_pos) + torch.pow(mu_pos, 2) - 1. - log_var_pos, 1)
    # KLD = torch.mean(F.mse_loss(log_var_pos, torch.ones_like(log_var_pos), reduction='none')
    #                  + F.mse_loss(mu_pos, torch.zeros_like(mu_pos), reduction='none'), dim=1)

    pos_bce = torch.mean(F.binary_cross_entropy(x_disc_pos, torch.zeros_like(x_disc_pos), reduction='none'), dim=1)
    neg_bce = 0
    if x_disc_neg is not None:
        neg_bce = torch.mean(F.binary_cross_entropy(x_disc_neg, torch.ones_like(x_disc_neg), reduction='none'), dim=1)
        # loss = pos_bce + MSE + KLD + neg_bce
        loss = pos_bce + MSE + KLD + neg_bce
    else:
        loss = pos_bce

    return loss, neg_bce, pos_bce, MSE


def evaluator(data_loader):
    model.eval()
    b_loss = []
    if model_name == "TranAD":
        loss_func = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for x, y in data_loader:
                x = x.permute(1, 0, 2).cuda()
                y = y.permute(1, 0, 2).cuda()
                z = model(x, y)
                if isinstance(z, tuple): z = z[1]
                l1 = loss_func(z, y)
                b_loss.append(torch.mean(l1[0], dim=1))
                # b_loss.append(torch.mean(loss, dim=1))
        return torch.cat(b_loss).detach().cpu().numpy()
    elif model_name == "VAE_MODEL":
        with torch.no_grad():
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                if target_dims is not None:
                    # y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(-1)
                # loss = loss_func(output[1], y)
                # loss = torch.squeeze(loss, 1)
                loss = vae_loss(x, *output)[0]
                # b_loss.append(torch.mean(torch.topk(loss, k=math.ceil(0.3 * out_dim), dim=1).values, dim=1))
                # b_loss.append(torch.mean(loss, dim=1))
                b_loss.append(loss)

        return torch.cat(b_loss).detach().cpu().numpy()
    elif model_name == "GDN":
        loss_func = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x.permute(0, 2, 1))
                if target_dims is not None:
                    # y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(-1)
                loss = loss_func(output, y.squeeze(1))
                # loss = torch.squeeze(loss, 1)
                # loss = vae_loss(x, *output)[0]
                # b_loss.append(torch.mean(torch.topk(loss, k=math.ceil(0.3 * out_dim), dim=1).values, dim=1))
                b_loss.append(torch.mean(loss, dim=1))

        return torch.cat(b_loss).detach().cpu().numpy()
    else:
        loss_func = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                if target_dims is not None:
                    # y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(-1)
                loss = loss_func(output[1], y)
                loss = torch.squeeze(loss, 1)
                # b_loss.append(torch.mean(torch.topk(loss, k=math.ceil(0.3 * out_dim), dim=1).values, dim=1))
                b_loss.append(torch.mean(loss, dim=1))
                # b_loss.append(loss)

        return torch.cat(b_loss).detach().cpu().numpy()


if __name__ == "__main__":
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()
    args.__setattr__("dataset", "ETL")
    args.__setattr__("lookback", 20)
    args.__setattr__("epochs", 50)
    # args.__setattr__("group", "2-1")
    model_name = ''
    # model_name = "TranAD"
    model_name = "VAE_MODEL"
    # model_name = "GDN"

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        heads = 2
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
        # (x_train, _), (x_test, y_test) = get_SMD_data()
    elif dataset in ['MSL', 'SMAP', 'SWaT', 'WADI']:
        if dataset == 'MSL':
            heads = 11
        else:
            heads = 5
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset.startswith('ETL'):
        heads = 4
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset.startswith('XSJ'):
        heads = 1
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # log_dir = f'{output_path}/logs'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    label = y_test[window_size:] if y_test is not None else None
    print(f"anomaly rate: {sum(label) / len(x_test)}")
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    target_dims = None
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    anomaly_data, sample_index = sample_anomaly_data(test_dataset, label, 0.1)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    train_start = time.time()
    if model_name == "TranAD":
        model = TranAD(n_features).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        loss_func = nn.MSELoss(reduction='none')
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            b_loss = []
            n = epoch + 1
            for x, y in train_loader:
                x = x.permute(1, 0, 2).cuda()
                y = y.permute(1, 0, 2).cuda()
                z = model(x, y)
                l1 = (1 / n) * loss_func(z[0], y) + (1 - 1 / n) * loss_func(z[1], y)
                loss = torch.mean(l1)
                b_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            b_loss = np.array(b_loss)
            avg_loss = np.sqrt((b_loss ** 2).mean())
            epoch_time = time.time() - epoch_start

            print(
                f"[Epoch {epoch + 1}] loss = {avg_loss:.5f} [{epoch_time:.1f}s]")
    elif model_name == "VAE_MODEL":
        # model = MTAD_Transform(n_features, window_size, out_dim).cuda()
        model = VAE_MODEL(n_features, window_size, out_dim).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        loss_func = nn.MSELoss(reduction='mean')
        loss_arr = []
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            b_loss = []
            mse_loss = []
            n = epoch + 1
            for x, y in train_loader:
                x = x.cuda()
                y = y.cuda()
                x_neg = random.choice(anomaly_data).cuda()
                optimizer.zero_grad()
                output = model(x, x_neg)
                if target_dims is not None:
                    # y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(-1)
                loss_a = vae_loss(x, *output)
                loss = torch.mean(loss_a[0])
                b_loss.append(loss.item())
                mse_loss.append(torch.mean(loss_a[3]).item())
                loss.backward()
                optimizer.step()
            # scheduler.step()

            b_loss = np.array(b_loss)
            avg_loss = b_loss.mean()
            avg_mse_loss = np.array(mse_loss).mean()
            loss_arr.append(avg_loss)
            epoch_time = time.time() - epoch_start

            print(
                f"[Epoch {epoch + 1}] loss = {avg_loss:.5f} mes_loss = {avg_mse_loss:.5f} [{epoch_time:.1f}s]")

    elif model_name == "GDN":
        edge_index = torch.cat((torch.arange(n_features).repeat(n_features).unsqueeze(0),
                                torch.arange(n_features).repeat_interleave(n_features).unsqueeze(0)),
                               dim=0).float().cuda()
        model = GDN([edge_index], n_features, input_dim=window_size, topk=6 if dataset == "ETL" else 20).cuda()
        # edge_index = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
        loss_func = nn.MSELoss(reduction='mean')
        loss_arr = []
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            b_loss = []
            mse_loss = []
            n = epoch + 1
            for x, y in train_loader:
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                output = model(x.permute(0, 2, 1))
                if target_dims is not None:
                    y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(1)
                # loss = (1 - n / n_epochs) * loss_func(output[0], x) + (n / n_epochs) * loss_func(output[1], y)
                # loss = (1 / n) * loss_func(output[0], x) + (1 - 1 / n) * loss_func(output[1], y)
                # loss = loss_func(output[0], x) + loss_func(output[1], y)
                loss = loss_func(output, y.squeeze(1))
                b_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            b_loss = np.array(b_loss)
            avg_loss = b_loss.mean()
            loss_arr.append(avg_loss)
            epoch_time = time.time() - epoch_start

            print(
                f"[Epoch {epoch + 1}] loss = {avg_loss:.5f}  [{epoch_time:.1f}s]")
    #     plt.plot(np.array(loss_arr), 'b', label='loss')
    #     plt.savefig(f"figure/{model.__class__.__name__}_{dataset}.jpg")

    else:
        model = MTAD_MODEL4(n_features, window_size, out_dim).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        # scheduler = StepLR(optimizer, step_size=3, gamma=0.2)
        loss_func = nn.MSELoss(reduction='mean')
        loss_arr = []
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            b_loss = []
            n = epoch + 1
            for x, y in train_loader:
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                output = model(x)
                if target_dims is not None:
                    # y = y[:, :, target_dims]
                    x = x[:, :, target_dims]
                #     output = output.squeeze(-1)
                # loss = (1 - n / n_epochs) * loss_func(output[0], x) + (n / n_epochs) * loss_func(output[1], y)
                # loss = (1 / n) * loss_func(output[0], x) + (1 - 1 / n) * loss_func(output[1], y)
                loss = loss_func(output[0], x) + loss_func(output[1], y)
                # loss = loss_func(output, y)
                # loss_a = vae_loss(x, *output)
                b_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()

            b_loss = np.array(b_loss)
            avg_loss = b_loss.mean()
            loss_arr.append(avg_loss)
            epoch_time = time.time() - epoch_start

            print(
                f"[Epoch {epoch + 1}] loss = {avg_loss:.5f} [{epoch_time:.1f}s]")
    print(f"train time: {time.time() - train_start}")
    test_loss = evaluator(test_loader)
    # train_loss = evaluator(train_loader)

    # test_loss, label = remove_sampled_data(test_loss, label, sample_index)

    # os.makedirs(f"output/{dataset}/loss", exist_ok=True)
    # np.save(f"output/{dataset}/loss/test_loss_{model.__class__.__name__}.npy", test_loss)
    # # np.save(f"output/{dataset}/loss/train_loss.npy", train_loss)
    # np.save(f"output/{dataset}/loss/label_{model.__class__.__name__}.npy", label)
    #
    torch.save(model, f"output/{dataset}/{model.__class__.__name__}.pt")
    np.save(f"output/{dataset}/sample_index_{model.__class__.__name__}.npy", sample_index)

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "ETL": 1, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "XSJ": 1, "SWaT": 1, "WADI": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    key = "ETL" if str(args.dataset).startswith("ETL") else key
    reg_level = reg_level_dict[key]
    # e_eval = epsilon_eval(train_loss, test_loss, label, reg_level)

    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.9091, 0.017),
        "ETL": (0.7945, 0.0272),
        "SMD": (0.9970, 0.001),
        # "SMD-2": (0.9925, 0.001),
        # "SMD-3": (0.9999, 0.001),
        "XSJ": (0.9999, 0.001),
        "SWaT": (0.9999, 0.001),
        "WADI": (0.9999, 0.001),
    }
    # key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    key = "ETL" if str(args.dataset).startswith("ETL") else key
    key = "SMD" if str(args.dataset).startswith("SMD") else key
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q
    # p_eval = pot_eval(train_loss, test_loss, label, q, level)

    # range_dict = {
    #     "SMAP": (0.01, 1),
    #     "ETL": (0.001, 0.2),
    #     "MSL": (0.001, 0.1),
    #     "SMD": (0.0001, 0.01),
    # }
    # range_dict = {
    #     "SMAP": (0.01, 1),
    #     "ETL2": (0.01, 0.2),
    #     "MSL": (0.001, 0.1),
    #     "SMD": (0.01, 0.5),
    #     "SWaT": (0.01, 2),
    #     "WADI": (0.01, 2),
    # }
    #
    # search_range = range_dict[dataset]
    # bf_eval = bf_search(test_loss, label, start=np.percentile(test_loss, 70), end=np.percentile(test_loss, 99),
    #                     step_num=100, verbose=False)
    bf_eval1 = bf_search1(test_loss, label, start=np.percentile(test_loss, 70), end=np.percentile(test_loss, 99),
                          step_num=100, verbose=False)

    # print(f"Results using epsilon method:\n {e_eval}")
    # print(f"Results using peak-over-threshold method:\n {p_eval}")
    # print(f"Results using best f1 score search:\n {bf_eval}")
    print(f"Results using best f1 score search:\n {bf_eval1}")
