#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as standard_transforms

#from linknet import LinkNet
#from linknet_scse import LinkNet
from linknet_scse2 import LinkNet
#from linknet_4apr2019 import LinkNet

resume_epoch = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
#torch.cuda.set_device(1)
#ckpt_path = 'ckpt_linknet_data10'
ckpt_path = 'ckpt_skipscse'
#ckpt_path = 'ckpt_4apr2019'
exp_name = 'TSHR-scse'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
args = {
    'num_class': 2,
    'ignore_label': 255,
    'num_gpus': 3,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 20,
    'lr': 0.0001,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'opt': 'adam',
}

class TSHRDataset(Dataset):
    def __init__(self, img_dir):
        self.img_anno_pairs = glob(img_dir)

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _target = Image.open(self.img_anno_pairs[index]).convert('L')
        _img = Image.open(self.img_anno_pairs[index][:-9] +'.png').convert('RGB')
        #print('img unique',np.unique(np.array(_img)))
        #print('target unique',np.unique(np.array(_target)))

        hflip = random.random() < 0.5
        if hflip:
            _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            _target = _target.transpose(Image.FLIP_LEFT_RIGHT)

        #_img = _img.resize((1024, 512), Image.BILINEAR)

        _img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
        #print("_img shape",_img.shape)
        _target = np.array(_target)
        _target[_target == 255] = 1
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


if __name__ == '__main__':

    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/train_v5/**_mask.png'
    img_dir='/media/mmlab/data/jiayi/data32_washeem_v6/train/**_mask.png'
    #img_dir='/media/mmlab/data/jiayi/data33_washeem_v7/train/**_mask.png' #take 3/10 for test
    #img_dir='/media/mmlab/data/jiayi/data34_washeem_v8/train/**_mask.png' #removed 5%,10%
    #img_dir='/media/mmlab/data/jiayi/data35_washeem_v9/train/**_mask.png'
    #img_dir='/media/mmlab/data/jiayi/data36_washeem_v10/train/**_mask.png'
    #img_dir = '/home/mmlab/jiayi/data19_dataset_patient1to5_old/train_v3/**_mask.png'
    
    dataset = TSHRDataset(img_dir=img_dir)
    print(dataset.__len__())
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    model = LinkNet(n_classes=2)
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    if args['opt'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=0.0002,
                              momentum=0.99, weight_decay=0.0001)
    elif args['opt'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args['lr'], weight_decay=0.0001)

    criterion = CrossEntropyLoss2d(size_average=True).cuda()
    model.train()
    epoch_iters = dataset.__len__() / args['batch_size']
    max_epoch = args['num_epoch']
    
    
    args['snapshot'] = 'epoch_' + str(resume_epoch) + '.pth.tar'
    model_path=os.path.join(ckpt_path, exp_name, args['snapshot'])
    #model.load_state_dict(torch.load(model_path))
    
    print('max_epoch: ',str(max_epoch))
    for epoch in range(resume_epoch+1, max_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print("\n outputs",outputs.shape,inputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
                    epoch, batch_idx + 1, epoch_iters, loss.item(),
                    optimizer.param_groups[0]['lr']))

        snapshot_name = 'epoch_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth.tar'))

