import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from AutoEncoder.AutoEncoder import AE_MODEL
from args import get_parser
from eval_methods import *
from utils import *


def ae_loss(x, x_mapper_pos, x_disc_pos, x_disc_neg=None, x_pos_rec=None, x_mapper_neg=None):
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
    MSE, Mapper_loss, neg_bce = 0, 0, 0
    pos_bce = torch.mean(F.binary_cross_entropy(x_disc_pos, torch.zeros_like(x_disc_pos), reduction='none'), dim=1)
    # MSE = torch.mean(F.mse_loss(x_pos_rec, x, reduction='none'), dim=(1, 2))
    if x_disc_neg is not None:
        MSE = torch.mean(F.mse_loss(x_pos_rec, x, reduction='none'), dim=(1, 2))
        Mapper_loss = torch.mean(F.mse_loss(x_mapper_pos, torch.ones_like(x_mapper_pos), reduction='none'), dim=1)
        neg_bce = torch.mean(F.binary_cross_entropy(x_disc_neg, torch.ones_like(x_disc_neg), reduction='none'), dim=1)


    if x_disc_neg is not None:

        loss = MSE + Mapper_loss + pos_bce + neg_bce
        # loss = MSE
    else:
        loss = pos_bce

    return loss, neg_bce, pos_bce, MSE, Mapper_loss


def evaluator(data_loader):
    model.eval()
    b_loss = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()
            output = model(x)
            if target_dims is not None:
                x = x[:, :, target_dims]
            loss = ae_loss(x, *output)[0]
            b_loss.append(loss)

    return torch.cat(b_loss).detach().cpu().numpy()


if __name__ == "__main__":
    id = datetime.now().strftime("%d%m%Y_%H%M%S")


    parser = get_parser()
    args = parser.parse_args()
    args.__setattr__("dataset", "SMAP")
    args.__setattr__("lookback", 20)
    args.__setattr__("epochs",  20)
    # args.__setattr__("group", "2-1")

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

    model = AE_MODEL(n_features, window_size, out_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
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
            x_neg = random.choice(anomaly_data).cuda()
            optimizer.zero_grad()
            output = model(x, x_neg)
            if target_dims is not None:
                x = x[:, :, target_dims]
            loss_a = ae_loss(x, *output)
            loss = torch.mean(loss_a[0])
            b_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # scheduler.step()

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

    os.makedirs(f"output/{dataset}/loss", exist_ok=True)
    np.save(f"output/{dataset}/loss/test_loss.npy", test_loss)
    # np.save(f"output/{dataset}/loss/train_loss.npy", train_loss)
    np.save(f"output/{dataset}/loss/label.npy", label)

    bf_eval = bf_search(test_loss, label, start=np.percentile(test_loss, 60), end=np.percentile(test_loss, 99), step_num=100, verbose=False)

    print(f"Results using best f1 score search:\n {bf_eval}")







