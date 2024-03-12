
import torch
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_scatter import scatter_mean
import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
import time
from tensorboardX import SummaryWriter
from Model.model import *


SEED = 0
np.random.random(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)


def classify(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize,
             dataname, iter, fold_count):
    unsup_model = Net(64, 3).to(device)

    for unsup_epoch in range(30):

        optimizer = th.optim.Adam(unsup_model.parameters(), lr=lr, weight_decay=weight_decay)
        # sheulder1 = CosineAnnealingLR(optimizer, T_max=2)
        unsup_model.train()
        traindata_list, _ = loadBiData(dataname, treeDic, x_train + x_test, x_test, 0.2, 0.2)
        # train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers=10)

        batch_idx = 0
        loss_all = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            optimizer.zero_grad()
            Batch_data = Batch_data.to(device)
            loss = unsup_model(Batch_data).to(device)
            loss_all += loss.item() * (max(Batch_data.batch) + 1)

            loss.backward()
            optimizer.step()
            # sheulder1.step()
            batch_idx = batch_idx + 1
        loss = loss_all / len(train_loader)
    name = "best_pre_" + dataname + "_4unsup" + ".pkl"
    th.save(unsup_model.state_dict(), name)
    print('Finished the unsuperivised training.', '  Loss:', loss)
    print("Start classify!!!")
    # unsup_model.eval()

    log_train = 'logs/' + datasetname + '/' + 'train' + 'iter_' + str(iter)
    writer_train = SummaryWriter(log_train)
    log_test = 'logs/' + datasetname + '/' + 'test' + 'iter_' + str(iter)
    writer_test = SummaryWriter(log_test)

    model = Classfier(64 * 3, 64, 4).to(device)
    opt = th.optim.Adam(model.parameters(), lr=0.0005, weight_decay=weight_decay)
    # sheulder2 = CosineAnnealingLR(opt, T_max=2)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        model.train()
        unsup_model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            _, Batch_embed = unsup_model.encoder(Batch_data.x, Batch_data.edge_index, Batch_data.batch)
            out_labels = model(Batch_embed, Batch_data)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            opt.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()
            # sheulder2.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,
                                                                                                               epoch,
                                                                                                               batch_idx,
                                                                                                               loss.item(),
                                                                                                               train_acc))
            batch_idx = batch_idx + 1

        writer_train.add_scalar('train_loss', np.mean(avg_loss), global_step=epoch + 1)
        writer_train.add_scalar('train_acc', np.mean(avg_acc), global_step=epoch + 1)
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        unsup_model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            Batch_embed = unsup_model.encoder.get_embeddings(Batch_data)
            val_out = model(Batch_embed, Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        writer_test.add_scalar('val_loss', np.mean(temp_val_losses), global_step=epoch + 1)
        writer_test.add_scalar('val_accs', np.mean(temp_val_accs), global_step=epoch + 1)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('unsup_epoch:', (unsup_epoch + 1), '   results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'RDEA_' + str(fold_count) + '_', dataname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if epoch >= 199:
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4

lr=0.0005
weight_decay=1e-4
patience=10
n_epochs=5
batchsize=16
datasetname="Weibo"
# modelname = sys.argv[1]
iterations=int(1)
model="BiGCN"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_accs,ACC1,ACC2,PRE1,PRE2,REC1,REC2,F1,F2 = [],[],[],[],[],[],[],[],[]
for iter in range(iterations):
    fold0_x_test, fold0_x_train,\
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train,  \
    fold3_x_test, fold3_x_train,  \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_model(0,treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               tddroprate,budroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,modelname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs,accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_model(1,treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               tddroprate,budroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,modelname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_model(2,treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               tddroprate,budroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,modelname,
                                                                                               iter)
    # train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(3,treeDic,
    #                                                                                            fold3_x_test,
    #                                                                                            fold3_x_train,
    #                                                                                            tddroprate,budroprate, lr,
    #                                                                                            weight_decay,
    #                                                                                            patience,
    #                                                                                                n_epochs,
    #                                                                                                batchsize,
    #                                                                                                datasetname,modelname,
    #                                                                                                iter)
    # train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(4,treeDic,
    #                                                                                            fold4_x_test,
    #                                                                                            fold4_x_train,
    #                                                                                            tddroprate,budroprate, lr,
    #                                                                                            weight_decay,
    #                                                                                            patience,
    #                                                                                            n_epochs,
    #                                                                                            batchsize,
    #                                                                                            datasetname,modelname,
    #                                                                                            iter)
    test_accs.append((accs_0+accs_1+accs_2)/3)
    ACC1.append((acc1_0 + acc1_1 + acc1_2) / 3)
    ACC2.append((acc2_0 + acc2_1 +acc2_2) / 3)
    PRE1.append((pre1_0 + pre1_1 + pre1_2) / 3)
    PRE2.append((pre2_0 + pre2_1 + pre2_2) / 3)
    REC1.append((rec1_0 + rec1_1 + rec1_2) / 3)
    REC2.append((rec2_0 + rec2_1 + rec2_2) / 3)
    F1.append((F1_0+F1_1+F1_2)/3)
    F2.append((F2_0 + F2_1 + F2_2) / 3)
print("weibo:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))

