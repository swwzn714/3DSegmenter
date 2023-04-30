import argparse
import os, sys, time
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch import nn
import glob
from mmseg.core import mean_iou
from network import create_segmenter

IGNORE_LABEL = 255
_SPLITTER = ','
parser = argparse.ArgumentParser()

parser.add_argument('--save', default='./log', help='folder to output model checkpoints')
parser.add_argument('--train_data', default='/media/wu/8TSSD/mergelabeldataset', help='folder of training files')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--patch_size', type=int, default=4, help='patch size')
parser.add_argument('--swin_depths', type=list, default=[2,2], help='swin depth, T:[2,2]; B:[2,4]; M:[3,6]; L:[4,8]')
parser.add_argument('--swin_heads', type=list, default=[3,3], help='swin heads, T:[3,3]; B:[3,3]; M:[4,4]; L:[6,6]')
parser.add_argument('--d_model', type=int, default=96, help='dim model, T:96; B:96; M:128; L:256')
parser.add_argument('--input_nf', type=int, default=4, help='input feature size')
parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--enable_val', type=bool, default=False, help='enable validation')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_lr', type=int, default=10, help='decay learning rate by half every n epochs')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument('--scheduler_step_size', type=int, default=0,help='#iters before scheduler step (0 for each epoch)')

class BlockDataset(Dataset):
    def __init__(self, datadir):
        self.filelist=glob.glob(os.path.join(datadir,"*.npz"))
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, index):
        data=np.load(self.filelist[index])
        targetsdf128loc = data["targetsdf128loc"]
        targetsdf128sdf = data["targetsdf128"]
        targetsdf128color = data["targetsdf128color"]
        semanticlabel = data["semanticlabel"]
        return targetsdf128loc, targetsdf128sdf, targetsdf128color, semanticlabel

def collate(batch):
    sdf128 = []
    color128 = []
    semantic128 = []
    for i in range(len(batch)):
        sdf = sparse_to_dense_np(batch[i][0], batch[i][1][:,np.newaxis], 128, 128, 128, default_val=-3)
        sdf128.append(sdf[np.newaxis, :, :, :])
        color = sparse_to_dense_np(batch[i][0], batch[i][2], 128, 128, 128, default_val=0)
        color128.append(color[np.newaxis, :, :, :])
        semantic = sparse_to_dense_np(batch[i][0], batch[i][3][:,np.newaxis], 128, 128, 128, default_val=0)
        semantic128.append(semantic[np.newaxis, :, :, :])
    sdf128 = np.concatenate(sdf128,axis=0)
    color128 = np.concatenate(color128,axis=0)
    semantic128 = np.concatenate(semantic128,axis=0)
    sdf128 = torch.from_numpy(sdf128).to(device)
    color128 = torch.from_numpy(color128).to(device).permute(0,4,1,2,3)
    color128 = color128* 2 - 1
    semantic128 = torch.from_numpy(semantic128).long().to(device)
    inputs = torch.concat([sdf128.unsqueeze(1),color128],dim=1)
    return inputs,semantic128


def print_log_info(epoch, iter, mean_train_losses, mean_train_ious,mean_val_losses, mean_val_iou, time, log):
    splitters = ['Epoch: ', ' iter: '] if log is None else ['', ',']
    values = [epoch, iter]
    values.extend(mean_train_losses)
    for h in range(len(mean_train_losses)):
        id = 'total' if h == 0 else str(h - 1)
        id = 'sdf' if h + 1 == len(mean_train_losses) else id
        if log is None:
            splitters.append(' loss_train(' + id + '): ')
        else:
            splitters.append(',')
    values.extend([mean_train_ious])
    if log is None:
        splitters.extend([' train_iou: '])
    else:
        splitters.extend([','])
    if mean_val_losses is not None:
        values.extend(mean_val_losses)
        for h in range(len(mean_val_losses)):
            id = 'total' if h == 0 else str(h - 1)
            id = 'sdf' if h + 1 == len(mean_val_losses) else id
            if log is None:
                splitters.append(' loss_val(' + id + '): ')
            else:
                splitters.append(',')
        values.extend([mean_val_iou])
        if log is None:
            splitters.extend([' val_iou: '])
        else:
            splitters.extend([','])

    else:
        splitters.extend([''] * (len(mean_train_losses) + 2))
        values.extend([''] * (len(mean_train_losses) + 2))
    values.append(time)
    if log is None:
        splitters.append(' time: ')
    else:
        splitters.append(',')
    info = ''
    for k in range(len(splitters)):
        if log is None and isinstance(values[k], float):
            info += splitters[k] + '{:.6f}'.format(values[k])
        else:
            info += splitters[k] + str(values[k])
    if log is None:
        print(info, file=sys.stdout)
    else:
        print(info, file=log)


def print_log(log, epoch, iter, train_losses, train_ious, val_losses, val_ious, time):
    train_losses = np.array(train_losses)
    train_ious = np.array(train_ious)
    mean_train_losses = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_losses]
    mean_train_ious = -1 if (len(train_ious) == 0 or np.all(train_ious < 0)) else np.mean(
        train_ious[train_ious >= 0])
    mean_val_losses = None
    mean_val_ious = None
    if val_losses:
        val_losses = np.array(val_losses)
        val_ious = np.array(val_ious)
        mean_val_losses = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_losses]
        mean_val_ious = -1 if (len(val_ious) == 0 or np.all(val_ious < 0)) else np.mean(
            val_ious[val_ious >= 0])
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious,mean_val_losses, mean_val_ious, time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious,mean_val_losses, mean_val_ious, time, log)
    else:
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious, None, None,time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious, None, None,time, log)
    log.flush()

def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimx, dimy, dimz, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense.reshape([dimx, dimy, dimz, nf_values])
    return dense.reshape([dimx, dimy, dimz])

def dump_args_txt(args, output_file):
    with open(output_file, 'w') as f:
        f.write('%s\n' % str(args))

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

def compute_iou(seg_pred,seg_gt):
    num = seg_pred.shape[0]
    seg_pred_np = seg_pred.detach().cpu().numpy()
    seg_gt_np = seg_gt.detach().cpu().numpy()
    list_seg_pred = np.split(seg_pred_np,indices_or_sections=num,axis=0)
    list_seg_gt = np.split(seg_gt_np,indices_or_sections=num,axis=0)
    ret_metrics = mean_iou(
        results=list_seg_pred,
        gt_seg_maps=list_seg_gt,
        num_classes=30,
        ignore_index=IGNORE_LABEL,
    )
    ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]]

    ret_metrics_mean = torch.tensor(
        [
            np.round(np.nanmean(ret_metric.astype(np.float32)) * 100, 2)
            for ret_metric in ret_metrics
        ],
        dtype=float,
        device=device,
    )
    pix_acc, mean_acc, miou = ret_metrics_mean
    return miou

args = parser.parse_args()
print(args)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_dataset = BlockDataset(os.path.join(args.train_data,"train"))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
val_dataset = BlockDataset(os.path.join(args.train_data,"val"))
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

net_kwargs = {'image_size': (128, 128, 128),
              'patch_size': args.patch_size,
              'd_model': args.d_model,
              'dropout': 0.0,
              'drop_path_rate': 0.1,
              'decoder': {'name': 'linear'},
              'n_cls': 30,
              'swin_depths': args.swin_depths,
              'swin_heads': args.swin_heads,
              'channels': args.input_nf
              }

model = create_segmenter(net_kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
last_epoch = -1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr, gamma=0.5, last_epoch=last_epoch)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
print('#params ', count_num_model_params(model))

def train(epoch, iter, dataloader, log_file):
    train_losses = [[]]
    train_iou = []
    model.train()
    start = time.time()

    if args.scheduler_step_size == 0:
        scheduler.step()
    for t, sample in enumerate(dataloader):
        inputs,label = sample
        output_sem = model(inputs)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        loss = criterion(output_sem, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses[0].append(loss.item())
        iter += 1
        if args.scheduler_step_size > 0 and iter % args.scheduler_step_size == 0:
            scheduler.step()
        if iter % 20 == 0:
            sem_pred=output_sem.argmax(1)
            train_iou.append(compute_iou(sem_pred,label).item())
            took = time.time() - start
            print_log(log_file, epoch, iter, train_losses, train_iou, None, None, took)
        if iter % 2000 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))
            else:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))

    return train_losses, train_iou, iter


def validate(epoch, iter, dataloader, log_file):
    val_losses = [[]]
    val_iou = []
    model.eval()
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs,label = sample
            output_sem = model(inputs)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
            loss = criterion(output_sem, label)
            val_losses[0].append(loss.item())
            sem_pred = output_sem.argmax(1)
            val_iou.append(compute_iou(sem_pred,label).item())
    return val_losses, val_iou



def main():
    _OVERFIT = False
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        input('warning: save dir %s exists, press key to delete and continue' % args.save)

    dump_args_txt(args, os.path.join(args.save, 'args.txt'))
    log_file = open(os.path.join(args.save, 'log.csv'), 'w')
    headers = ['epoch', 'iter', 'train_loss(total)']
    headers.extend(['train_iou'])
    headers.extend(['time'])
    log_file.write(_SPLITTER.join(headers) + '\n')
    log_file.flush()

    has_val = args.enable_val
    log_file_val = None
    if has_val:
        headers = headers[:-1]
        headers.append('val_loss(total)')
        headers.extend(['val_iou'])
        headers.extend(['time'])
        log_file_val = open(os.path.join(args.save, 'log_val.csv'), 'w')
        log_file_val.write(_SPLITTER.join(headers) + '\n')
        log_file_val.flush()
    # start training
    print('starting training...')
    iter = args.start_epoch * (len(train_dataset) // args.batch_size)
    bestepoch_iou = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()

        train_losses, train_iou, iter = train(epoch, iter, train_dataloader,log_file)
        if has_val:
            val_losses, val_iou, = validate(epoch, iter, val_dataloader,log_file_val)

        took = time.time() - start
        if has_val:
            print_log(log_file_val, epoch, iter, train_losses, train_iou, val_losses, val_iou, took)
        else:
            print_log(log_file, epoch, iter, train_losses, train_iou, None, None, took)
        if torch.cuda.device_count() > 1:
            torch.save({'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
        else:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
        if has_val:
            val_iou = np.array(val_iou)
            currentepoch_iou = np.mean(val_iou)
            if bestepoch_iou<currentepoch_iou:
                print('update best model in epoch %s' % str(epoch + 1))
                bestepoch_iou = currentepoch_iou
                if torch.cuda.device_count() > 1:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                        os.path.join(args.save, 'best-model.pth'))
                else:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        os.path.join(args.save, 'best-model.pth'))
        else:
            train_iou = np.array(train_iou)
            currentepoch_iou = np.mean(train_iou)
            if bestepoch_iou < currentepoch_iou:
                print('update best model in epoch %s' % str(epoch + 1))
                bestepoch_iou = currentepoch_iou
                if torch.cuda.device_count() > 1:
                    torch.save(
                        {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),'optimizer': optimizer.state_dict()},
                        os.path.join(args.save, 'best-model.pth'))
                else:
                    torch.save(
                        {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        os.path.join(args.save, 'best-model.pth'))
    log_file.close()
    if has_val:
        log_file_val.close()


if __name__ == '__main__':
    main()


