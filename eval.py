import argparse
import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from mmseg.core import eval_metrics
import tqdm
from network import create_segmenter

IGNORE_LABEL = 255
parser = argparse.ArgumentParser()

parser.add_argument('--room_dir', default='Path to PanoRooms', help='path to panorooms')
parser.add_argument('--checkpoint_dir', default='3DSeg-T.pth', help='path to saved checkpoint')
parser.add_argument('--patch_size', type=int, default=4, help='patch size')
parser.add_argument('--swin_depths', type=list, default=[2,2], help='swin depth, T:[2,2]; B:[2,4]; M:[3,6]; L:[4,8]')
parser.add_argument('--swin_heads', type=list, default=[3,3], help='swin heads, T:[3,3]; B:[3,3]; M:[4,4]; L:[6,6]')
parser.add_argument('--d_model', type=int, default=96, help='dim model, T:96; B:96; M:128; L:256')
parser.add_argument('--input_nf', type=int, default=4, help='input feature size')

def create_label_mapping():
    labelmergemappping = np.zeros(99, dtype=np.uint8)

    ceilings = np.array((1, 4, 30, 31, 37, 52, 53), dtype=np.uint8)
    labelmergemappping[ceilings] = 1

    cabinets = np.array((2, 6, 28, 47, 48, 55, 69, 74, 91, 22, 76, 23, 56), dtype=np.uint8)
    labelmergemappping[cabinets] = 2

    tables = np.array((3, 12, 36, 68, 72, 80, 81), dtype=np.uint8)
    labelmergemappping[tables] = 3

    sewerpipes = np.array((5), dtype=np.uint8)
    labelmergemappping[sewerpipes] = 4

    doors = np.array((7, 98, 21, 44, 58), dtype=np.uint8)
    labelmergemappping[doors] = 5

    lamps = np.array((8, 32, 19, 33), dtype=np.uint8)
    labelmergemappping[lamps] = 6

    sofas = np.array((9, 10, 18, 46, 89, 96), dtype=np.uint8)
    labelmergemappping[sofas] = 7

    chairs = np.array((14, 16, 50, 71, 78, 83), dtype=np.uint8)
    labelmergemappping[chairs] = 8

    stools = np.array((25, 35, 94), dtype=np.uint8)
    labelmergemappping[stools] = 9

    appliances = np.array((11, 34, 86, 65), dtype=np.uint8)
    labelmergemappping[appliances] = 10

    buildelements = np.array((13), dtype=np.uint8)
    labelmergemappping[buildelements] = 11

    others = np.array((15, 26), dtype=np.uint8)
    labelmergemappping[others] = 12

    beds = np.array((17, 20, 67, 70, 87, 88), dtype=np.uint8)
    labelmergemappping[beds] = 13

    customizedplatforms = np.array((27, 43, 45, 66, 90, 54, 61), dtype=np.uint8)
    labelmergemappping[customizedplatforms] = 14

    plants = np.array((29), dtype=np.uint8)
    labelmergemappping[plants] = 15

    baseboards = np.array((38), dtype=np.uint8)
    labelmergemappping[baseboards] = 16

    walls = np.array((40, 49, 57, 75, 93, 95, 60, 85, 39, 24), dtype=np.uint8)
    labelmergemappping[walls] = 17

    basins = np.array((41), dtype=np.uint8)
    labelmergemappping[basins] = 18

    baths = np.array((42), dtype=np.uint8)
    labelmergemappping[baths] = 19

    floors = np.array((51, 62, 79), dtype=np.uint8)
    labelmergemappping[floors] = 20

    arts = np.array((59), dtype=np.uint8)
    labelmergemappping[arts] = 21

    beams = np.array((63), dtype=np.uint8)
    labelmergemappping[beams] = 22

    stairs = np.array((64), dtype=np.uint8)
    labelmergemappping[stairs] = 23

    flues = np.array((73), dtype=np.uint8)
    labelmergemappping[flues] = 24

    recreations = np.array((77), dtype=np.uint8)
    labelmergemappping[recreations] = 25

    columns = np.array((82), dtype=np.uint8)
    labelmergemappping[columns] = 26

    wardrobes = np.array((84), dtype=np.uint8)
    labelmergemappping[wardrobes] = 27

    mirrors = np.array((92), dtype=np.uint8)
    labelmergemappping[mirrors] = 28

    customized_wainscot = np.array((97), dtype=np.uint8)
    labelmergemappping[customized_wainscot] = 29

    return labelmergemappping

class SceneDataset(Dataset):
    def __init__(self, data_dir, test_scenes):
        roomsdir = data_dir
        f = open(test_scenes, "r")
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

def torgb(rgb_vals):
    _color_const = 256 * 256
    inputshape = rgb_vals.shape
    targetcolor = np.zeros((inputshape[0],inputshape[1],inputshape[2],3),dtype=np.float32)
    colors_b = np.floor(rgb_vals / _color_const)
    colors_g = np.floor((rgb_vals - colors_b * _color_const) / 256)
    colors_r = rgb_vals - colors_b * _color_const - colors_g * 256
    targetcolor[:, :, :, 0]= colors_r
    targetcolor[:, :, :, 1]= colors_g
    targetcolor[:, :, :, 2]= colors_b
    targetcolor = targetcolor/255
    return targetcolor

def compute_metrics(seg_pred,seg_gt):
    num = seg_pred.shape[0]
    seg_pred_np = seg_pred.detach().cpu().numpy()
    seg_gt_np = seg_gt.detach().cpu().numpy()
    list_seg_pred = np.split(seg_pred_np,indices_or_sections=num,axis=0)
    list_seg_gt = np.split(seg_gt_np,indices_or_sections=num,axis=0)
    ret_metrics = eval_metrics(
        results=list_seg_pred,
        gt_seg_maps=list_seg_gt,
        num_classes=30,
        ignore_index=IGNORE_LABEL,
        metrics=['mIoU', 'mDice', 'mFscore'],
    )
    ret_metrics = [ret_metrics["aAcc"], ret_metrics["IoU"], ret_metrics["Acc"],ret_metrics["Dice"],
                   ret_metrics["Fscore"],ret_metrics["Precision"],ret_metrics["Recall"]]
    ret_metrics_mean = torch.tensor(
        [
            np.round(np.nanmean(ret_metric.astype(np.float32)) * 100, 2)
            for ret_metric in ret_metrics
        ],
        dtype=float,
        device=device,
    )
    aAcc,IoU,Acc,Dice,Fscore,Precision,Recall = ret_metrics_mean
    return aAcc,IoU,Acc,Dice,Fscore,Precision,Recall


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    args = parser.parse_args()
    print(args)
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
    model.to(device)

    test_dataset = SceneDataset(args.room_dir,"test.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    aAccs = []
    IoUs = []
    Accs = []
    Dices = []
    Fscores = []
    Precisions = []
    Recalls = []
    weight = []
    with torch.no_grad():
        for sample in tqdm.tqdm(test_dataloader):
            inputs, semanticlabel = sample
            blockshape = np.asarray(inputs.shape)[2:] / 128
            blocknumber = blockshape[0] * blockshape[1] * blockshape[2]
            output_sem = model(inputs)
            sem_pred = output_sem.argmax(1)

            aAcc,IoU,Acc,Dice,Fscore,Precision,Recall = compute_metrics(sem_pred, semanticlabel)
            aAccs.append(aAcc.item())
            IoUs.append(IoU.item())
            Accs.append(Acc.item())
            Dices.append(Dice.item())
            Fscores.append(Fscore.item())
            Precisions.append(Precision.item())
            Recalls.append(Recall.item())
            weight.append(blocknumber)
            currentaAcc = np.dot(aAccs, weight) / np.sum(np.asarray(weight))
            currentIoU = np.dot(IoUs, weight) / np.sum(np.asarray(weight))
            currentAcc = np.dot(Accs, weight) / np.sum(np.asarray(weight))
            currentDice = np.dot(Dices, weight) / np.sum(np.asarray(weight))
            currentFscore = np.dot(Fscores, weight) / np.sum(np.asarray(weight))
            currentPrecision = np.dot(Precisions, weight) / np.sum(np.asarray(weight))
            currentRecall = np.dot(Recalls, weight) / np.sum(np.asarray(weight))
        print("aAcc",round(currentaAcc,2),"IoU",round(currentIoU,2),
              "Acc",round(currentAcc,2),"Dice",round(currentDice,2),
              "Fscore",round(currentFscore,2),"Precision",round(currentPrecision,2),
              "Recall",round(currentRecall,2))


if __name__ == '__main__':
    main()


