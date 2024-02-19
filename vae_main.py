import math
import time
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from eval_methods import *
from AE.ae_model import ED

from args import get_parser
from utils import *
from my_model import MTAD_MODEL, MTAD_MODEL2, MTAD_MODEL3, MTAD_MODEL4
from TranAD.models import TranAD

def normalize_array(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
    return arr_normalized


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
    else:
        loss_func = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                # if target_dims is not None:
                #     y = y[:, :, target_dims].squeeze(-1)
                #     output = output.squeeze(-1)
                loss = loss_func(output[1], y)
                loss = torch.squeeze(loss, 1)
                b_loss.append(torch.mean(torch.topk(loss, k=math.ceil(0.3 * n_features), dim=1).values, dim=1))
                # b_loss.append(torch.mean(loss, dim=1))

        return torch.cat(b_loss).detach().cpu().numpy()

if __name__ == "__main__":
    id = datetime.now().strftime("%d%m%Y_%H%M%S")


    parser = get_parser()
    args = parser.parse_args()
    args.__setattr__("dataset", "MSL")
    args.__setattr__("lookback", 20)
    args.__setattr__("epochs",  50)
    # args.__setattr__("group", "3-2")
    model_name = ''
    # model_name = "TranAD"

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

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    label = y_test[window_size:] if y_test is not None else None
    print(f"anomaly rate: {sum(label) / len(x_test)}")
    n_features = x_train.shape[1]

    # target_dims = get_target_dims(dataset)
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
    sampled_anomaly_data = sample_anomaly_data(test_dataset, label)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    train_start = time.time()

    model = ED(n_features, window_size, out_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    loss_func = nn.MSELoss(reduction='mean')
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
            # if target_dims is not None:
            #     y = y[:, :, target_dims].squeeze(-1)
            #     output = output.squeeze(-1)
            # loss = (1 - n / n_epochs) * loss_func(output[0], x) + (n / n_epochs) * loss_func(output[1], y)
            loss = loss_func(output[0], x) + loss_func(output[1], y)
            # loss = loss_func(output, y)
            b_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        b_loss = np.array(b_loss)
        avg_loss = b_loss.mean()
        epoch_time = time.time() - epoch_start

        print(
            f"[Epoch {epoch + 1}] loss = {avg_loss:.5f} [{epoch_time:.1f}s]")

    print(f"train time: {time.time() - train_start}")
    test_loss = evaluator(test_loader)
    # train_loss = evaluator(train_loader)

    # os.makedirs(f"output/{dataset}/loss", exist_ok=True)
    # np.save(f"output/{dataset}/loss/test_loss.npy", test_loss)
    # np.save(f"output/{dataset}/loss/train_loss.npy", train_loss)
    # np.save(f"output/{dataset}/loss/label.npy", label)

    # Some suggestions for Epsilon args
    # reg_level_dict = {"SMAP": 0, "MSL": 0, "ETL": 1, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "XSJ": 1,"SWaT": 1,"WADI": 1 }
    # key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    # key = "ETL" if str(args.dataset).startswith("ETL") else key
    # reg_level = reg_level_dict[key]
    # e_eval = epsilon_eval(train_loss, test_loss, label, reg_level)
    #
    # level_q_dict = {
    #     "SMAP": (0.90, 0.005),
    #     "MSL": (0.9091, 0.017),
    #     "ETL": (0.7945, 0.0272),
    #     "SMD": (0.9970, 0.001),
    #     # "SMD-2": (0.9925, 0.001),
    #     # "SMD-3": (0.9999, 0.001),
    #     "XSJ": (0.9999, 0.001),
    #     "SWaT": (0.9999, 0.001),
    #     "WADI": (0.9999, 0.001),
    # }
    # # key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    # key = "ETL" if str(args.dataset).startswith("ETL") else key
    # key = "SMD" if str(args.dataset).startswith("SMD") else key
    # level, q = level_q_dict[key]
    # if args.level is not None:
    #     level = args.level
    # if args.q is not None:
    #     q = args.q
    # p_eval = pot_eval(train_loss, test_loss, label, q, level)
    #
    # # range_dict = {
    # #     "SMAP": (0.01, 1),
    # #     "ETL": (0.001, 0.2),
    # #     "MSL": (0.001, 0.1),
    # #     "SMD": (0.0001, 0.01),
    # # }
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
    bf_eval = bf_search(test_loss, label, start=np.percentile(test_loss, 50), end=np.percentile(test_loss, 99), step_num=200, verbose=False)
    bf_eval1 = bf_search1(test_loss, label, start=np.percentile(test_loss, 70), end=np.percentile(test_loss, 99),
                          step_num=100, verbose=False)

    # print(f"Results using epsilon method:\n {e_eval}")
    # print(f"Results using peak-over-threshold method:\n {p_eval}")
    print(f"Results using best f1 score search:\n {bf_eval}")
    print(f"Results using best f1 score1 search:\n {bf_eval1}")







