import argparse
import os, sys, time
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import glob
import torch.nn.functional as F
from mmseg.core import mean_iou
from teachermodel import load_pano_model,STATS
import torchvision.transforms.functional as transF
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_rotation,
)
from pytorch3d.structures import Volumes
import cv2
import matplotlib.pyplot as plt
from eval import create_label_mapping,torgb
from network import create_segmenter
import math

IGNORE_LABEL = 255
_SPLITTER = ','
parser = argparse.ArgumentParser()

parser.add_argument('--save', default='./log', help='folder to output model checkpoints')
parser.add_argument('--room_dir', default='Path to PanoRooms', help='folder of rooms')
parser.add_argument('--pano_dir', default='Path to Panoramas', help='folder of panoramas')
parser.add_argument('--checkpoint_student', default='3DSeg-T.pth', help='3d student checkpoint')
parser.add_argument('--checkpoint_teacher', default='segmenter2d_base/checkpoint.pth', help='2d teacher checkpoint')
parser.add_argument('--split_val', default='val.txt', help='validation split')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--patch_size', type=int, default=4, help='patch size')
parser.add_argument('--swin_depths', type=list, default=[2,2], help='swin depth, T:[2,2]; B:[2,4]; M:[3,6]; L:[4,8]')
parser.add_argument('--swin_heads', type=list, default=[3,3], help='swin heads, T:[3,3]; B:[3,3]; M:[4,4]; L:[6,6]')
parser.add_argument('--d_model', type=int, default=96, help='dim model, T:96; B:96; M:128; L:256')
parser.add_argument('--input_nf', type=int, default=4, help='input feature size')
parser.add_argument('--max_epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--enable_val', type=bool, default=False, help='enable validation')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument('--T', type=float, default=20, help='temperature for distillation')
parser.add_argument('--alpha', type=float, default=1.0, help='weight parameter')


class DistillDataset(Dataset):
    def __init__(self, data_dir,pano_dir):
        annotationdir = os.path.join(pano_dir,"annotations","training")
        imagedir = os.path.join(pano_dir,"images","training")
        self.annotation_list = glob.glob(os.path.join(annotationdir,"*.png"))
        self.image_list = []
        self.room_list = []
        for i in range(len(self.annotation_list)):
            scenename = os.path.splitext(os.path.split(self.annotation_list[i])[1])[0]
            self.image_list.append(os.path.join(imagedir, scenename + ".jpg"))
            self.room_list.append(os.path.join(data_dir,scenename+".npz"))
        self.label = create_label_mapping()
    def __len__(self):
        return len(self.room_list)
    def __getitem__(self, index):
        vol_file = np.load(self.room_list[index])
        sdf = -3 * vol_file["sdf128"]
        color = vol_file["color128"]
        semanticlabel = vol_file["semanticlabel"]
        color = torgb(color)
        newcolor = np.zeros_like(color)
        newcsem = np.zeros_like(semanticlabel)
        locs = np.where(sdf>-3)
        newcolor[locs[0],locs[1],locs[2],:]= color[locs[0],locs[1],locs[2],:]
        newcsem[locs[0],locs[1],locs[2]] = semanticlabel[locs[0],locs[1],locs[2]]
        color = 2 * newcolor - 1
        sdf = torch.from_numpy(sdf).to(device).unsqueeze(-1)
        color = torch.from_numpy(color).to(device)
        newcsem = self.label[newcsem]
        semanticlabel = torch.from_numpy(newcsem).to(device).long()
        inputs = torch.concat([sdf,color],dim=-1).permute(3,0,1,2)

        anno2d = cv2.imread(self.annotation_list[index])[:,:,0]
        im2d = cv2.imread(self.image_list[index]).astype(np.float32)/255
        anno2d = torch.from_numpy(anno2d).to(device).long()
        im2d = torch.from_numpy(im2d).to(device)
        return  inputs,semanticlabel,im2d,anno2d

class ValidateonRooms(Dataset):
    def __init__(self, data_dir, split_val):
        roomsdir = data_dir
        f = open(split_val, "r")
        scenelist = f.read().split("\n")[:-1]
        f.close()
        self.filelist = []
        for scene in scenelist:
            self.filelist.append(os.path.join(roomsdir,scene + ".npz"))
        self.label = create_label_mapping()
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, index):
        vol_file = np.load(self.filelist[index])
        sdf = -3 * vol_file["sdf128"]
        color = vol_file["color128"]
        semanticlabel = vol_file["semanticlabel"]
        color = torgb(color)
        newcolor = np.zeros_like(color)
        newcsem = np.zeros_like(semanticlabel)
        locs = np.where(sdf > -3)
        newcolor[locs[0], locs[1], locs[2], :] = color[locs[0], locs[1], locs[2], :]
        newcsem[locs[0], locs[1], locs[2]] = semanticlabel[locs[0], locs[1], locs[2]]
        color = 2 * newcolor - 1
        sdf = torch.from_numpy(sdf).to(device).unsqueeze(-1)
        color = torch.from_numpy(color).to(device)
        newcsem = self.label[newcsem]
        semanticlabel = torch.from_numpy(newcsem).to(device).long()
        inputs = torch.concat([sdf, color], dim=-1).permute(3, 0, 1, 2)
        return inputs, semanticlabel

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

def print_log_info(epoch, iter, mean_train_losses, mean_train_ious,mean_val_losses, mean_val_ious, time, log):
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
        values.extend([mean_val_ious])
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



def print_log(log, epoch, iter, train_losses, train_ious,train_weights, val_losses, val_ious, val_weights, time):
    train_losses = np.array(train_losses)
    train_ious = np.array(train_ious)
    train_weights = np.array(train_weights)
    mean_train_losses = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_losses]
    mean_train_ious = np.dot(train_ious, train_weights) / np.sum(train_weights)
    mean_val_losses = None
    mean_val_ious = None
    if val_losses:
        val_losses = np.array(val_losses)
        val_ious = np.array(val_ious)
        val_weights = np.array(val_weights)
        mean_val_losses = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_losses]
        mean_val_ious = np.dot(val_ious, val_weights) / np.sum(val_weights)
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious,
                       mean_val_losses, mean_val_ious, time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious,
                       mean_val_losses, mean_val_ious,  time, log)
    else:
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious, None, None,
                       time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_ious, None, None,
                       time, log)
    log.flush()


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

def _equirect_facetype(h: int, w: int) -> torch.Tensor:
    """0F 1R 2B 3L 4U 5D"""

    int_dtype = torch.int64

    tp = torch.roll(
        torch.arange(4)  # 1
        .repeat_interleave(w // 4)  # 2 same as np.repeat
        .unsqueeze(0)
        .transpose(0, 1)  # 3
        .repeat(1, h)  # 4
        .view(-1, h)  # 5
        .transpose(0, 1),  # 6
        shifts=3 * w // 8,
        dims=1,
    )

    # Prepare ceil mask
    mask = torch.zeros((h, w // 4), dtype=torch.bool)
    idx = torch.linspace(-math.pi, math.pi, w // 4) / 4
    idx = h // 2 - torch.round(torch.atan(torch.cos(idx)) * h / math.pi)
    idx = idx.type(int_dtype)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = torch.roll(torch.cat([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[torch.flip(mask, dims=(0,))] = 5

    return tp.type(int_dtype)

def create_equi_grid(
    h_out: int,
    w_out: int,
    w_face: int,
    batch: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    theta = torch.linspace(
        -math.pi, math.pi, steps=w_out, dtype=dtype, device=device
    )
    phi = (
        torch.linspace(
            math.pi, -math.pi, steps=h_out, dtype=dtype, device=device
        )
        / 2
    )
    phi, theta = torch.meshgrid([phi, theta])

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)

    # xy coordinate map
    coor_x = torch.zeros((h_out, w_out), dtype=dtype, device=device)
    coor_y = torch.zeros((h_out, w_out), dtype=dtype, device=device)

    # FIXME: there's a bug where left section (3L) has artifacts
    # on top and bottom
    # It might have to do with 4U or 5D
    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * torch.tan(theta[mask] - math.pi * i / 2)
            coor_y[mask] = (
                -0.5
                * torch.tan(phi[mask])
                / torch.cos(theta[mask] - math.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * torch.tan(math.pi / 2 - phi[mask])
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = c * torch.cos(theta[mask])
        elif i == 5:
            c = 0.5 * torch.tan(math.pi / 2 - torch.abs(phi[mask]))
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = -c * torch.cos(theta[mask])

    # Final renormalize
    coor_x = torch.clamp(
        torch.clamp(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1
    )
    coor_y = torch.clamp(
        torch.clamp(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1
    )

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    # repeat batch
    coor_x = coor_x.repeat(batch, 1, 1)
    coor_y = coor_y.repeat(batch, 1, 1)

    grid = torch.stack((coor_y, coor_x), dim=-3).to(device)
    return grid

def cube2equi(horizon,height=512,width=1024,mode='bilinear'):
    bs, c, w_face, _ = horizon.shape
    horizon_device = horizon.device
    grid = create_equi_grid(
        h_out=height,
        w_out=width,
        w_face=w_face,
        batch=bs,
        dtype=torch.float32,
        device=horizon_device,
    )
    _, _, h, w = horizon.shape
    grid = grid.permute(0, 2, 3, 1)
    norm_uj = torch.clamp(2 * grid[..., 0] / (h - 1) - 1, -1, 1)
    norm_ui = torch.clamp(2 * grid[..., 1] / (w - 1) - 1, -1, 1)
    grid[..., 0] = norm_ui
    grid[..., 1] = norm_uj
    out = F.grid_sample(
        horizon,
        grid,
        mode=mode,
        align_corners=True,
        padding_mode="reflection",
    )
    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out

def render(sdf,feature,render_size=512,output_h=512, output_w=1024):
    eps = 1e-8
    featuredim = feature.shape[1]
    campos = torch.zeros((6, 3), dtype=torch.float32)
    at = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], dtype=torch.float32)
    T = torch.zeros((6, 3), dtype=torch.float32)
    up = torch.tensor([[1, eps, 0]], dtype=torch.float32).repeat(6, 1)
    R = look_at_rotation(camera_position=campos, at=at, up=up)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=90.0)
    raysampler = NDCMultinomialRaysampler(
        image_width=render_size,
        image_height=render_size,
        n_pts_per_ray=100,
        min_depth=0.1,
        max_depth=7.0,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    renderer = VolumeRenderer(
        raysampler=raysampler, raymarcher=raymarcher,
    )
    render_images_list=[]
    volumes = Volumes(densities=sdf, features=feature, voxel_size=0.04)
    for i in range(6):
        rendered_images, rendered_silhouettes =\
            renderer(cameras=cameras[i], volumes=volumes)[0].split([featuredim, 1],dim=-1)
        render_images_list.append(rendered_images)
    rendered_images = torch.concat(render_images_list,dim=0)
    temp = rendered_images[4].clone().transpose(0,1).flip(dims=[1])
    rendered_images[4,:] = temp
    rendered_images[5] = torch.rot90(rendered_images[5], k=3)
    panolist = rendered_images.permute(1, 0, 2, 3).reshape(render_size, -1, featuredim)
    equ = cube2equi(horizon=panolist.permute(2, 0, 1)[None, ...],
                    height=output_h,
                    width=output_w,
                    mode='bilinear'
                    )
    return equ



args = parser.parse_args()
print(args)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

train_dataset = DistillDataset(args.room_dir,args.pano_dir)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = ValidateonRooms(args.room_dir,args.split_val)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

#load 3d student model
model = create_segmenter(net_kwargs)
checkpoint = torch.load(args.checkpoint_student)
model.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=list(range(0,8,2)),gamma=0.5,last_epoch=-1)
model.to(device)
print('#params ', count_num_model_params(model))

#load 2d teacher model
model_path = args.checkpoint_teacher
teachermodel, variant = load_pano_model(model_path)
teachermodel.to(device)
for p in teachermodel.parameters():
    p.requires_grad = False
teachermodel.eval()
normalization_name = variant["dataset_kwargs"]["normalization"]
normalization = STATS[normalization_name]


def train(epoch, iter, dataloader, log_file):
    train_losses = [[]]
    train_iou = []
    train_weights = []
    model.train()
    start = time.time()

    for t, sample in enumerate(dataloader):
        inputs, semantic, im2d, anno2d = sample
        blockshape = np.asarray(inputs.shape)[2:] // 128
        weight = blockshape[0] * blockshape[1] * blockshape[2]
        output_sem= model(inputs)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        loss = criterion(output_sem, semantic)
        sdf = (inputs[:, :1, :, :, :] + 3) / 6
        student_segmap = render(sdf=sdf, feature=output_sem,render_size=256, output_h=512, output_w=1024)
        student_segmap = student_segmap.unsqueeze(0)

        with torch.no_grad():
            im = im2d.squeeze().permute(2,0,1)
            im = transF.normalize(im, normalization["mean"], normalization["std"])
            im = im.unsqueeze(0)
            teacher_segmap = teachermodel(im)

        soft_distillation_loss = F.kl_div(
            F.log_softmax(student_segmap / args.T, dim=1),
            F.log_softmax(teacher_segmap / args.T, dim=1),
            reduction='sum',
            log_target=True
        ) * (args.T * args.T) / student_segmap.numel()
        loss = loss + soft_distillation_loss * args.alpha

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses[0].append(loss.item())
        iter += 1

        if iter % 1 == 0:
            sem_pred = output_sem.argmax(1)
            train_iou.append(compute_iou(sem_pred, semantic).item())
            train_weights.append(weight)
        if iter % 20 == 0:
            took = time.time() - start
            print_log(log_file, epoch, iter, train_losses, train_iou, train_weights, None, None, None, took)
        if iter % 1500 == 0:
            return train_losses, train_iou, train_weights, iter


def validation(epoch, iter, dataloader, log_file):
    val_losses = [[]]
    val_iou = []
    val_weights = []
    model.eval()
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs,semantic128 = sample
            blockshape = np.asarray(inputs.shape)[2:] // 128
            blocknumber = blockshape[0] * blockshape[1] * blockshape[2]
            output_sem = model(inputs)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
            loss = criterion(output_sem, semantic128)
            val_losses[0].append(loss.item())
            sem_pred = output_sem.argmax(1)
            val_iou.append(compute_iou(sem_pred,semantic128).item())
            val_weights.append(blocknumber)
    return val_losses, val_iou, val_weights



def main():
    _OVERFIT = False
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        input('warning: save dir %s exists, press key to delete and continue' % args.save)

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
    bestepoch = -1
    bestepoch_iou = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()
        train_losses, train_iou, train_weights ,iter = train(epoch, iter, train_dataloader,log_file)
        if has_val:
            val_losses, val_iou, val_weights = validation(epoch, iter, val_dataloader, log_file_val)

        took = time.time() - start
        if has_val:
            print_log(log_file_val, epoch, iter, train_losses, train_iou, train_weights, val_losses, val_iou, val_weights, took)
        else:
            print_log(log_file, epoch, iter, train_losses, train_iou,train_weights, None, None, None, took)
        if torch.cuda.device_count() > 1:
            torch.save({'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
        else:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
        if has_val:
            val_iou = np.array(val_iou)
            val_weights = np.array(val_weights)
            currentepoch_iou = np.dot(val_iou,val_weights)/np.sum(val_weights)
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
            train_weights = np.array(train_weights)
            currentepoch_iou = np.dot(train_iou,train_weights)/np.sum(train_weights)
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
        scheduler.step()
        print("current lr:",optimizer.param_groups[0]['lr'])
    log_file.close()
    if has_val:
        log_file_val.close()


if __name__ == '__main__':
    main()


