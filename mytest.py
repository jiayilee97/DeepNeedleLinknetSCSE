# System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import time
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
from skimage.io import imread, imsave
# Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as standard_transforms
from torchvision.models import resnet18
import cv2
import math
import shutil
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from rms_angle_v6 import postprocess as rms_angle_error

parser = argparse.ArgumentParser()
parser.add_argument('linknetMode', type=str, help='scse/default/skipscse')
argsParsed = parser.parse_args()
linknetMode = argsParsed.linknetMode

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args = {
    'snapshot': '',
    'num_class': 2,
    'batch_size': 4,  # original:8
    'num_gpus': 1,
}

#linknetMode='default'
#linknetMode='scse'
#linknetMode='skipscse' 

if linknetMode=='linknet':
    from linknet import LinkNet as LinkNet
    ckpt_path = './ckpt_linknet' #best epoch 111
    args['exp_name']= 'TSHR-LinkNet'
    start_epoch=111 
    end_epoch=112
    pred_dir='predsLinknet/'
elif linknetMode=='scse':
    from linknet_scse1 import LinkNet #best epoch 61
    ckpt_path = './ckpt_scse_newData'
    args['exp_name']= 'TSHR-scse'
    start_epoch=76 #best 76
    end_epoch=77 #best 77
    pred_dir='predsScse/'
elif linknetMode=='skipscse':
    from linknet_scse2 import LinkNet
    ckpt_path = './ckpt_skipscse' #best epoch 133
    args['exp_name']= 'TSHR-scse'
    start_epoch=133 
    end_epoch=134
    pred_dir='predsSkipScse/'
    
print('linknetMode',linknetMode)

def CalculateAveragePrecision(rec, prec):
    #print('rec',rec)
    zippedData=np.array(list(zip(rec,prec)))
    unzippedData=np.array(sorted(zippedData,key=lambda s:s[0]))
    recSorted=unzippedData[:,0]
    precSorted=unzippedData[:,1]
    
    #print('rec2',rec,prec)
    
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in recSorted]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in precSorted]
    mpre.append(0)
    
    #print('mpre',mpre[:5],mrec[:5])
    #print('sort',np.array(sorted(zippedData,key=lambda s:s[0])))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    
    
    #return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
    #print('ap',ap)
    return ap

def getF1(precision,recall):
    F1=2*precision*recall/(precision+recall)
    
    return F1

def getIOU(prediction,target):
    # iou = tp/(fp+tp+fn)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    #intersection=intersection.astype(np.uint8)
    #union=union.astype(np.uint8)
    
    iou_score = np.sum(intersection) / np.sum(union)
    if np.isnan(iou_score)==True:
        iou_score=0
    
    return iou_score
    
def getRecall(pred,label):
    # Recall = tp/(tp+fn)
    
    tp = np.sum(np.logical_and(label, pred))
    fn = np.sum(np.logical_and(label, tp))
    recall = tp/(tp+fn)
    #print('d',recall,tp,fn)
    if np.isnan(recall)==True:
        recall=0
        #print('isnan',recall)
    
    return recall

def getPrecision(pred,label):
    # Precision = tp/(tp+fp)
    tp = np.sum(np.logical_and(label, pred))
    fp = np.sum(np.logical_and(tp, pred))
    precision = tp/(tp+fp)
    #print('p',precision,tp,fp,np.isnan(precision))
    if np.isnan(precision)==True:
        precision=0
        #print('isnan',precision)

    return precision

def getWeightedError(rmsAngle,rmsSlen,rmsTipDistError,imDiagLength):
    return rmsAngle/3/180 + rmsSlen/3/imDiagLength + rmsTipDistError/3/imDiagLength

def HausdorffDist(A,B):
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return (dH)

def dice(pred, label):
    
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_val

def specificity(TP, TN, FP, FN):
    return TN / (FP + TN)


def sensitivity(TP, TN, FP, FN):
    return TP / (TP + FN)

def spec_sens(pred, gt):
    # pred[pred>0] = 1
    # gt[gt>0] = 1
    A = np.logical_and(pred, gt)
    TP = float(A[A > 0].shape[0])
    TN = float(A[A == 0].shape[0])
    B = img_pred - labels
    FP = float(B[B > 0].shape[0])
    FN = float(B[B < 0].shape[0])
    specificity = TN / (FP + TN)
    sensitivity = TP / (TP + FN)
    #print(specificity, sensitivity)
    return specificity, sensitivity
    
# Get Dice, Hausdorff, Specificity, Sensitivity (DHSS) New
def getDHSSNew(labs, labelX,img_predX,mdice,mhausdorff,mspecificity,msensitivity):
    
    for instru_idx in range(1, len(labs)):
        labels_temp = np.zeros(labelX.shape)
        img_pred_temp = np.zeros(labelX.shape)
        labels_temp[labelX == labs[instru_idx]] = 1
        img_pred_temp[img_predX == labs[instru_idx]] = 1
        #print(labs[instru_idx],np.unique(img_pred[dice_idx]) )
        if (np.max(labels_temp) == 0):# or (np.max(img_pred_temp)==0):
            continue
        
        # visualise inputs to dice function
        diceX=dice(img_pred_temp, labels_temp)
        #print('m',labs[instru_idx],instru_idx)
        mdice[labs[instru_idx]].append(diceX)
        mhausdorff[labs[instru_idx]].append(directed_hausdorff(img_pred_temp, labels_temp)[0])
        spec, sens = spec_sens(img_pred_temp, labels_temp)
        mspecificity[labs[instru_idx]].append(spec)
        msensitivity[labs[instru_idx]].append(sens)
    return mdice,mhausdorff,mspecificity,msensitivity

def getAvgDiceHaus(mdice,mhausdorff,mspecificity,msensitivity):
    avg_dice = []
    avg_hd = []
    avg_spec = []
    avg_sens = []
    for idx_eval in range(1, args['num_class']):
        if idx_eval == 5 or idx_eval == 6 or idx_eval == 7 or math.isnan(float(np.mean(mdice[idx_eval]))):
            mdice[idx_eval] = 0
            continue
        #print(idx_eval, ':', np.mean(mdice[idx_eval]))
        avg_dice.append(np.mean(mdice[idx_eval]))
        avg_hd.append(np.mean(mhausdorff[idx_eval]))
        avg_spec.append(np.mean(mspecificity[idx_eval]))
        avg_sens.append(np.mean(msensitivity[idx_eval]))
    return avg_dice,avg_hd,avg_spec,avg_sens


def getUltrasoundBorders(img):
    #img=cv2.imread('filteredIMagesIOU/p2_3_003.png')
    #imgOri=img.copy()
    #cv2.imshow('img',img)
    blur = cv2.bilateralFilter(img,9,10,10)
    #cv2.imshow('blur',blur)    
    #cv2.imshow('blur1',blur)
    #img=blur
    blur[blur>=5]=255
    blur[blur<5]=0
    #blur[blur!=0]=255
    
    blur=cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(blur,127,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    c = max(contours, key = cv2.contourArea)
    
    mask=np.zeros(img.shape)
    cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    #cv2.imshow('filled',mask) # best results
    mask = mask.astype(np.uint8)
    return mask

def analyseDiceStats(mdice,printStats=True):
    #print('x',mdice)
    stats=[0]*10
    for dice in mdice:
        tmp=int(dice*10)
        #print('t',tmp)
        #tmp=str(tmp)
        stats[tmp]+=1
    if printStats is True:
        print('total',len(mdice))
        print('dice stat (ascend.)',stats)
        print('dice stat (%)',np.array(stats)/len(mdice))
        print('mean(before)',np.mean(mdice))
        mdiceSorted=sorted(mdice)[96:]
        print('worst',mdiceSorted[0])
    #print('mean(after)',np.mean(mdiceSorted))

def analyseIOUStats(mIOU,printStats=True):
    #print('x',mdice)
    stats=[0]*10
    for IOU in mIOU:
        tmp=int(IOU*10)
        #print('t',tmp)
        #tmp=str(tmp)
        stats[tmp]+=1
    if printStats is True:
        print('total',len(mIOU))
        print('IOU stat (ascend.)',stats)
        print('IOU stat (%)',np.array(stats)/len(mIOU))
        print('mean(before)',np.mean(mIOU))
        mIOUSorted=sorted(mIOU)[70:75]
        print('worst',mIOUSorted)
    #print('mean(after)',np.mean(mdiceSorted))

def patchImg(img):
    #img=cv2.imread('/media/leejiayi/DATA/ubuntu/scripts/sampleEdgeDetection.png')
    blurX=cv2.blur(img,(20,20))
    x=10
    tmp= np.array(img[:,:,0]<=x) & np.array(img[:,:,1]<=x) & np.array(img[:,:,2]<=x)
    tmp=tmp.astype(np.uint8)
    tmp=tmp*255
    tmp=cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('ddd',tmp)
    img2=img.copy()
    img2[tmp==255]=127
    tmp=blurX & tmp
    img2=tmp | img
    #cv2.imshow('img2',img2)
    #cv2.imwrite('patchedImg.png',img2)
    #cv2.waitKey(0)
    return img2

def getVisual(predVis,labelVis):
    print('\r\n')
    print('#'*20,'VISUAL_START','#'*20)
    #####---Code Begins---#####
    
    print('keys',predVis.keys(),labelVis.keys())

    # sort keys
    predKeys=sorted(predVis.keys())
    labelKeys=sorted(labelVis.keys())
    
    tmp=predKeys[0]
    predKeys[0:9]=predKeys[1:10]
    predKeys[9]=tmp
    
    tmp=labelKeys[0]
    labelKeys[0:9]=labelKeys[1:10]
    labelKeys[9]=tmp
    
    
    # further reorder predKeys and labelKeys (Mobarak's suggestion)
    predKeys[4]=predKeys[8]
    labelKeys[4]=labelKeys[8]

    # patch holes in image
    predVis[predKeys[0]]=patchImg(predVis[predKeys[0]])
    
    print('lb',labelKeys)
    print('pr',predKeys)

    
    COL=len(predKeys) - 6 # need to minus 1 to exclude B1_usBorder
    ROW=2

    imH=440
    imW=500
    pad=30
    canvasH=pad+(imH+pad)*ROW
    canvasW=pad+(imW+pad)*COL
    canvas=np.ones((canvasH,canvasW,3))
    canvas*=255

    for rowx in range(ROW):
        for colx in range(COL):
            if rowx==0:
                imVis=predVis
                imKeys=predKeys
            elif rowx==1:
                imVis=labelVis
                imKeys=labelKeys
                colx=max(colx,1)            
            # load image
            img=imVis[imKeys[colx]]
            #print('img',img.shape,np.unique(img))
            #print('col',colx)
            #img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            startX=pad+(imW+pad)*colx
            startY=pad+(imH+pad)*rowx
            endX=startX+imW
            endY=startY+imH
            #print('g',startX,endX,startY,endY)
            
            # force grayscale images to become BGR
            if len(img.shape)==2: 
                img=img.astype(np.uint8)
                print('x',np.unique(img))
                img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                
            canvas[startY:endY,startX:endX,:]=img

    # add usBorder
    '''
    rect=np.ones((imH+pad,canvasW,3))*255
    canvas=np.concatenate((rect,canvas),axis=0)
    colOffset=1
    rowOffset=0
    startX=pad+(imW+pad)*colOffset
    startY=pad+(imH+pad)*rowOffset
    endX=startX+imW
    endY=startY+imH
    canvas[startY:endY,startX:endX,:]=predVis[predKeys[-1]]
    '''
    
    # resize 
    #canvas = cv2.resize(canvas, None, fx=0.5,fy=0.5)

    #cv2.imshow('x',canvas)
    #cv2.waitKey(0)
    #cv2.imwrite('samplePredsOverlaid/wanted/postprocessed.png',canvas)

    # move image slightly lower
    startX=pad
    startY=int(pad+(imH+pad)/2)
    endX=startX+imW
    endY=startY+imH
    canvas[startY:endY,startX:endX,:]= canvas[10:450,10:510,:]
    canvas[10:startY,10:endX,:]=255
    
    
    # change BGR to RGB for plt
    tmp=canvas[:,:,0]
    canvas[:,:,0]=canvas[:,:,2]
    canvas[:,:,2]=tmp
    canvas=canvas.astype(np.uint8)
    
    # save image
    fig, ax = plt.subplots()
    
    # show image
    plt.axis("off")
    plt.imshow(canvas)
    plt.show()
    plt.draw()
    
    # Do the plot code
    fig.savefig('samplePredsOverlaid/wanted/postprocessed.png', format='png', dpi=600)

    
    #####---Code Ends---#####
    print('#'*20,'VISUAL_END','#'*20)
    print('\r\n')
    
def assignImOverlayKeys(img_uint8,label_uint8,rawImg):
    imOverlay=dict()
    key='pred'
    imOverlay[key]=img_uint8.copy()
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='rect'
    imOverlay[key]=predMetrics['imgRect']
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='extend'
    imOverlay[key]=predMetrics['imgNewDice'].copy()
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='predrect&extend'
    imOverlay[key]=predMetrics['imgNewDice'].copy()
    imOverlay['tmp']=predMetrics['imgRect']
    imOverlay[key]=cv2.addWeighted(imOverlay['tmp'],1,imOverlay[key],1,0)
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='labelextend'
    imOverlay[key]=labelMetrics['imgRect']
    imOverlay[key]=cv2.addWeighted(label_uint8,1,imOverlay[key],1,0)
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='labelraw'
    imOverlay[key]=label_uint8.copy()
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    key='labelrect&extend'
    imOverlay[key]=labelMetrics['imgNewDice']
    imOverlay['tmp']=labelMetrics['imgRect']
    imOverlay[key]=cv2.addWeighted(imOverlay['tmp'],1,imOverlay[key],1,0)
    imOverlay[key][:,:,0]=0
    imOverlay[key]=cv2.addWeighted(rawImg,1,imOverlay[key],1,0)
    return imOverlay

class TSHRDataset_test(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_anno_pairs = glob(img_dir)
        #print(type(self.img_anno_pairs))
        for index in range(len(self.img_anno_pairs)):
            _img = Image.open(self.img_anno_pairs[index][:-9] + '.png').convert('RGB')
            
    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _target = Image.open(self.img_anno_pairs[index]).convert('L')
        _img = Image.open(self.img_anno_pairs[index][:-9] + '.png').convert('RGB')
        #_img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
        _target = np.array(_target)
        
        
        
        _target[_target == 255] = 1
        _target = torch.from_numpy(np.array(_target)).long()
        #_img = _img.resize((512, 256), Image.BILINEAR)
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        return _img, _target, self.img_anno_pairs[index]

if __name__ == '__main__':
    #img_dir = '/media/mobarak/data/Datasets/TUMOR_SUR_HR/Test/img_seg/**[0-9000].jpg'
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5/**_mask.png'
    img_dir='/media/mmlab/data/jiayi/data32_washeem_v6/test/**_mask.png' #new dataset
    img_dir='filteredImagesIOU/**_mask.png' #new dataset (removed low dice frames)
    #img_dir='/media/mmlab/data/jiayi/data33_washeem_v7/test/**_mask.png'
    #img_dir='/media/mmlab/data/jiayi/data33_washeem_v7/train/**_mask.png'
    #img_dir='/media/mmlab/data/jiayi/data34_washeem_v8/test/**_mask.png' 
    #img_dir='/media/mmlab/data/jiayi/data35_washeem_v9/test/**_mask.png' 
    #img_dir='/media/mmlab/data/jiayi/data36_washeem_v10/test/**_mask.png'
    #img_dir='/media/mmlab/data/jiayi/data37_washeem_v11/test/**_mask.png' 
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_reduced2/**_mask.png' #linknet 80% accuracy dataset
    
    dataset = TSHRDataset_test(img_dir=img_dir)
    print("len",len(dataset))
    
    
    test_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2)
    #model = get_model("pspnet", n_classes=2)
    model = LinkNet(n_classes=2)
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    Best_Dice = 0
    Best_epoch=0
    Best_Dice2 = 0
    Best_epoch2=0
    Best_DiceRect=0
    Best_epochDiceRect=0
    Best_AvePrec=0
    Best_epochAvePrec=0
    bestIOUNoPP=0
    bestEpochIOUNoPP=0
    bestIOU=0
    bestEpochIOU=0
    
    
    bestError={
    'angle':5000,
    'slen':5000,
    'tipDist':5000,
    'weightedError':5000,
    'epochAngle':1,
    'epochSlen':1,
    'epochTip':1,
    'epochWeightedError':1
    }
    ppImgCombined=[]
    
    for epochs in range(start_epoch,end_epoch):
        args['snapshot'] = 'epoch_' + str(epochs) + '.pth.tar'
#        model.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))

        model_path=os.path.join(ckpt_path, args['exp_name'], args['snapshot'])

        model.load_state_dict(torch.load(model_path))
        #model.load_state_dict(torch.load('/home/mmlab/jiayi/pytorch/ckpt/TSHR-LinkNet/epoch_93_best.pth.tar'))
        model.eval()
        
        w, h = 0, args['num_class']
        
        # old dice
        mdice = []
        mspecificity = []
        msensitivity = []
        mhausdorff = []
        haus = []
        
        mdice = [[0 for x in range(w)] for y in range(h)]
        mspecificity = [[0 for x in range(w)] for y in range(h)]
        msensitivity = [[0 for x in range(w)] for y in range(h)]
        mhausdorff = [[0 for x in range(w)] for y in range(h)]
        haus = []
        
        # new dice
        mdice2 = [[0 for x in range(w)] for y in range(h)]
        mspecificity2 = [[0 for x in range(w)] for y in range(h)]
        msensitivity2 = [[0 for x in range(w)] for y in range(h)]
        mhausdorff2 = [[0 for x in range(w)] for y in range(h)]  
        skippedImg=0
        numBlankPred=0
        
        # dice of boundedRect mask
        mdiceRect=[]
        avgdiceRect=[]
        
        # list for recall, precision
        rec=[]
        prec=[]
        listIOU=[]
        listF1=[]
        listIOUNoPP=[]
        
        mytime = []
        mymin = 10000

        diceThres=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        squaredErrors={
            'angle':[],
            'slen':[],
            'tipDist':[]
        }
        
        # store Images & Masks with good dice
        filteredImgs={
            'imgs':[],
            'masks':[],
            'preds':[],
            'imnames':[],
            'predRect':[],
            'labelRect':[]
        }
        
        #imOverlay=dict()
        
        for batch_idx, data in enumerate(test_loader):
                # remove these lines afterwards
                '''
                if batch_idx==0:
                    continue
                elif batch_idx>1:
                    break
                '''
                 
                inputs, labels, mpath = data
                inputs = Variable(inputs).cuda()
                t0 = time.time()
                outputs = model(inputs)
                t1 = time.time()
                mytime.append((t1 - t0))
                img_pred = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
                labels = np.array(labels)
                
                inputsNumpy=inputs.cpu().numpy().astype(np.uint8)
                inputsNumpy=inputsNumpy.transpose(0, 2, 3, 1)
                
                for dice_idx in range(0,img_pred.shape[0]):
                #for dice_idx in range(1,2): #change back to line above
                    #print('dice',dice_idx)
                    if(np.max(labels[dice_idx])==0):
                        continue
                    
                    mdice \
                    ,mhausdorff \
                    ,mspecificity \
                    ,msensitivity \
                    =getDHSSNew(np.unique(labels[dice_idx]) \
                                ,labels[dice_idx] \
                                ,img_pred[dice_idx] \
                                ,mdice,mhausdorff \
                                ,mspecificity \
                                ,msensitivity)
                    
                    
                    #added on 27 May 2019    
                    img_pred[dice_idx][img_pred[dice_idx] == 1] = 255
                    labels[dice_idx][labels[dice_idx] == 1] = 255
                    save_image_flag=False
                    if save_image_flag == True:
                        seg_path = pred_dir+os.path.basename(mpath[dice_idx])

                        if not os.path.exists(pred_dir):
                            #os.mkdir(pred_dir)
                            print('created ',pred_dir)
                        for x in [5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]:
                            pred_dir_sub=pred_dir+str(x)
                            if not os.path.exists(pred_dir_sub):
                                #os.mkdir(pred_dir_sub)
                                print('created',pred_dir_sub)
                        
                        imsave(seg_path, img_pred[dice_idx])
                    
                    #if len(np.unique(img_pred[dice_idx])) !=1:
                        #print('iii',np.unique(img_pred[dice_idx]))
                    
                    #get angle and shortest length errors
                    pred_path=pred_dir+"/*_mask.png"
                    img_uint8=img_pred[dice_idx].astype(np.uint8)
                    img_uint8=cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                    label_uint8=labels[dice_idx].astype(np.uint8)
                    label_uint8=cv2.cvtColor(label_uint8, cv2.COLOR_GRAY2BGR)
                    predMetrics=dict()
                    labelMetrics=dict()
                    
                    # get ultrasound borders
                    usBorderMask=getUltrasoundBorders(inputsNumpy[dice_idx])

                        
                    (predMetrics['angle'],
                     predMetrics['slen'],
                     predMetrics['img'],
                     predMetrics['tipDist'],
                     predMetrics['imgNewDice'],
                     predMetrics['imgRect'],
                     predMetrics['imgRectNoPP'],
                     predMetrics['imVisualise']
                     )=rms_angle_error(img_uint8,usBorderMask,bboxmode='nofill')
                    #)=rms_angle_error(img_uint8,usBorderMask,bboxmode='fill')
                    
                    (labelMetrics['angle'],
                     labelMetrics['slen'],
                     labelMetrics['img'],
                     labelMetrics['tipDist'],
                     labelMetrics['imgNewDice'],
                     labelMetrics['imgRect'],
                     labelMetrics['imgRectNoPP'],
                     labelMetrics['imVisualise']
                     )=rms_angle_error(label_uint8,usBorderMask,bboxmode='nofill')
                    #)=rms_angle_error(label_uint8,usBorderMask,bboxmode='fill')
                    
                    
                    getVisualFlag=False
                    if getVisualFlag is True:
                        # add input image to visualise
                        predMetrics['imVisualise']['A1_inputIm'] \
                        =inputsNumpy[dice_idx].copy()
                        
                        # add blank image as labelMetrics['imVisualise'] 
                        # has 1 less image than predMetrics['imVisualise']
                        labelMetrics['imVisualise']['A1_inputIm'] \
                        =np.zeros(inputsNumpy[dice_idx].shape)
                        getVisual(predMetrics['imVisualise'],labelMetrics['imVisualise'])
                        #raise KeyboardInterrupt
                    
                    predMetrics['imgNewDice']=predMetrics['imgNewDice'].astype(np.uint8)
                    labelMetrics['imgNewDice']=labelMetrics['imgNewDice'].astype(np.uint8)
                    predMetrics['imgRect']=predMetrics['imgRect'].astype(np.uint8)
                    labelMetrics['imgRect']=labelMetrics['imgRect'].astype(np.uint8)
                    predMetrics['imgRectNoPP']=predMetrics['imgRectNoPP'].astype(np.uint8)
                    labelMetrics['imgRectNoPP']=labelMetrics['imgRectNoPP'].astype(np.uint8)
                
                    '''
                    cv2.imshow('pred',predMetrics['imgRect'])
                    cv2.imshow('label',labelMetrics['imgRect'])
                    if cv2.waitKey(0)==ord('q'):
                        raise KeyboardInterrupt
                    '''
                    
                    imOverlay = assignImOverlayKeys(img_uint8,label_uint8,inputsNumpy[dice_idx])
                    
                    #cv2.imshow('imgrect&extend',imOverlay['rect&extend']) #yes
                    #cv2.imshow('labelrect&extend',imOverlay['labelrect&extend'])#yes
                    #cv2.imshow('pred',imOverlay[key])
                    
                    pred_dir=os.path.join('samplePredsOverlaid/',linknetMode)
                    save_image_flag=False
                    if save_image_flag is True:

                        if not os.path.exists(pred_dir):
                            os.makedirs(pred_dir)
                            print('created ',pred_dir)
                        
                        key='predrect&extend'
                        seg_path = os.path.join(pred_dir,os.path.basename(mpath[dice_idx]))
                        seg_path = seg_path.replace('mask',key)
                        imsave(seg_path, imOverlay[key])
                        print('segpath',seg_path)
                        previousKey=key
                                                    
                        key='labelrect&extend'
                        seg_path = seg_path.replace(previousKey,key)
                        imsave(seg_path, imOverlay[key])
                        #print('segpath',seg_path)
                        previousKey=key

                        '''
                        key='labelrect&extend'
                        seg_path = seg_path.replace(previousKey,key)
                        #imsave(seg_path, imOverlay[key])
                        print('segpath',seg_path)
                        '''
                    
                    
                    #if ord('q') == cv2.waitKey(0):
                    #    raise KeyboardInterrupt
                    
                    
                    '''
                    try:
                        predMetrics['angle'],predMetrics['slen'],_,predMetrics['tipDist']=rms_angle_error(img_uint8)
                        labelMetrics['angle'],labelMetrics['slen'],_,labelMetrics['tipDist']=rms_angle_error(label_uint8)
                    except Exception as e:
                        print("type error: " + str(e))
                        skippedImg+=1
                        
                        continue
                    '''
                    shouldCombine=False
                    if shouldCombine is True:
                        ppImgCombined=np.concatenate((predMetrics['img'],labelMetrics['img']),axis=1)
                        cv2.imshow('y',ppImgCombined)
                        cv2.waitKey(1)
                        
                    squaredAngleError=(predMetrics['angle']-labelMetrics['angle'])**2
                    squaredSlen=(predMetrics['slen']-labelMetrics['slen'])**2
                    squaredTipDistError=(predMetrics['tipDist']-labelMetrics['tipDist'])**2
                    
                    if np.isnan(squaredAngleError):
                        squaredAngleError=0
                        #print('nan',squaredAngleError)
                    if np.isnan(squaredSlen):
                        squaredSlen=0
                        #print('nan',squaredSlen)
                    if np.isnan(squaredTipDistError):
                        squaredTipDistError=0
                        #print('nan',squaredSlen)
                    squaredErrors['angle'].append(squaredAngleError)
                    squaredErrors['slen'].append(squaredSlen)
                    squaredErrors['tipDist'].append(squaredTipDistError)
                    
                    # get dice, hausdorff, specificity, sensitivity
                    #mdice,mhausdorff,mspecificity,msensitivity=getDHSSNew(np.unique(labels[dice_idx]), labels[dice_idx],img_pred[dice_idx],mdice,mhausdorff,mspecificity,msensitivity)
                    #predMetrics['imgNewDice']=predMetrics['imgNewDice'].astype(np.uint8)
                    #labelMetrics['imgNewDice']=labelMetrics['imgNewDice'].astype(np.uint8)
                    
                    
                    #cv2.imshow('predExtend',predMetrics['imgNewDice'])
                    #cv2.imshow('labelExtend',labelMetrics['imgNewDice'])
                    #if ord('q') == cv2.waitKey(0):
                    #    raise KeyboardInterrupt
                    
                    predMetrics['imgNewDice']=cv2.cvtColor(predMetrics['imgNewDice'], cv2.COLOR_BGR2GRAY)
                    labelMetrics['imgNewDice']=cv2.cvtColor(labelMetrics['imgNewDice'], cv2.COLOR_BGR2GRAY)
                    predMetrics['imgNewDice'][predMetrics['imgNewDice'] == 255] = 1
                    labelMetrics['imgNewDice'][labelMetrics['imgNewDice'] == 255] = 1
                    
                    
                    
                    # get dice of bounded rect mask
                    predMetrics['imgRect'][predMetrics['imgRect'] == 255] = 1
                    labelMetrics['imgRect'][labelMetrics['imgRect'] == 255] = 1
                    diceRect=dice(predMetrics['imgRect'],labelMetrics['imgRect'])
                    mdiceRect.append(diceRect)
                    '''
                    # filter images with dice > 10%
                    if diceRect>=0.1:
                        #print('type',type(img_uint8[0,0]),np.unique(img_uint8))
                        imRectFilter=predMetrics['imgRect']
                        labelRectFilter=labelMetrics['imgRect']
                        imRectFilter[imRectFilter == 1] = 255
                        labelRectFilter[labelRectFilter == 1] = 255
                        imRectFilter=imRectFilter.astype(np.uint8)
                        labelRectFilter=labelRectFilter.astype(np.uint8)
                        #print('ssxx',np.unique(np.array(imRectFilter)),np.unique(np.array(labelRectFilter)))
                        filteredImgs['preds'].append(img_uint8)
                        filteredImgs['masks'].append(label_uint8)   
                        filteredImgs['imnames'].append(os.path.basename(mpath[dice_idx]))
                        filteredImgs['predRect'].append(imRectFilter)   
                        filteredImgs['labelRect'].append(labelRectFilter)
                        #print('imnames',os.path.basename(mpath[dice_idx]))
                        #print('xx',np.unique(img_pred[dice_idx]),np.unique(labels[dice_idx]))
                    '''
                        
                    
                    mdice2 \
                    ,mhausdorff2 \
                    ,mspecificity2 \
                    ,msensitivity2 \
                    =getDHSSNew(np.unique(labelMetrics['imgNewDice']) \
                                ,labelMetrics['imgNewDice'] \
                                ,predMetrics['imgNewDice'] \
                                ,mdice2,mhausdorff2 \
                                ,mspecificity2 \
                                ,msensitivity2)
                    
                    #print('mdice2',mdice2,np.array(mdice2).shape)
                    recX=getRecall(predMetrics['imgRect'],labelMetrics['imgRect'])
                    precX=getPrecision(predMetrics['imgRect'],labelMetrics['imgRect'])
                    rec.append(recX)
                    prec.append(precX)
                    IOU=getIOU(predMetrics['imgRect'],labelMetrics['imgRect'])
                    listIOU.append(IOU)
                    #listF1.append(getF1(precX,recX))
                    
                    
                    # get IOU for pred (before postprocessing)
                    predMetrics['imgRectNoPP'][predMetrics['imgRectNoPP'] == 255] = 1
                    labelMetrics['imgRectNoPP'][labelMetrics['imgRectNoPP'] == 255] = 1
                    listIOUNoPP.append(getIOU(predMetrics['imgRectNoPP'],labelMetrics['imgRectNoPP']))
                    #print('iou nopp',getIOU(predMetrics['imgRectNoPP'] \
                    #                          ,labelMetrics['imgRectNoPP']))
                                            
                    
                    # filter images with IOU > 10%
                    if IOU>=0.1:
                        #print('type',type(img_uint8[0,0]),np.unique(img_uint8))
                        imRectFilter=predMetrics['imgRect']
                        labelRectFilter=labelMetrics['imgRect']
                        imRectFilter[imRectFilter == 1] = 255
                        labelRectFilter[labelRectFilter == 1] = 255
                        imRectFilter=imRectFilter.astype(np.uint8)
                        labelRectFilter=labelRectFilter.astype(np.uint8)
                        #print('ssxx',np.unique(np.array(imRectFilter)),np.unique(np.array(labelRectFilter)))
                        filteredImgs['preds'].append(img_uint8)
                        filteredImgs['masks'].append(label_uint8)   
                        filteredImgs['imnames'].append(os.path.basename(mpath[dice_idx]))
                        filteredImgs['predRect'].append(imRectFilter)   
                        filteredImgs['labelRect'].append(labelRectFilter)
                        #print('imnames',os.path.basename(mpath[dice_idx]))
                        #print('xx',np.unique(img_pred[dice_idx]),np.unique(labels[dice_idx]))
                        
                        
                    
                    
        meanAvePrec=CalculateAveragePrecision(rec, prec)
                
        avg_dice,avg_hd,avg_spec,avg_sens=getAvgDiceHaus(mdice,mhausdorff,mspecificity,msensitivity)
        avg_dice2,avg_hd2,avg_spec2,avg_sens2=getAvgDiceHaus(mdice2,mhausdorff2,mspecificity2,msensitivity2)
        
        rmsAngle=math.sqrt(np.mean(squaredErrors['angle']))
        rmsSlen=math.sqrt(np.mean(squaredErrors['slen']))
        rmsTipDistError=math.sqrt(np.mean(squaredErrors['tipDist']))
        imDiagLen=math.sqrt(img_pred[dice_idx].shape[0]**2+img_pred[dice_idx].shape[1]**2)
        weightedError=getWeightedError(rmsAngle \
                                        ,rmsSlen \
                                        ,rmsTipDistError, \
                                        imDiagLen)
        if rmsAngle<bestError['angle']:
            bestError['angle']=rmsAngle
            bestError['epochAngle']=epochs
        if rmsSlen<bestError['slen']:
            bestError['slen']=rmsSlen
            bestError['epochSlen']=epochs
        if rmsTipDistError<bestError['tipDist']:
            bestError['tipDist']=rmsTipDistError
            bestError['epochTip']=epochs
        if weightedError<bestError['weightedError']:
            bestError['weightedError']=weightedError
            bestError['epochWeightedError']=epochs
        if np.mean(avg_dice) > Best_Dice:
            Best_Dice = np.mean(avg_dice)
            Best_epoch = epochs
        if np.mean(avg_dice2) > Best_Dice2:
            Best_Dice2 = np.mean(avg_dice2)
            Best_epoch2 = epochs
        if np.mean(mdiceRect) > Best_DiceRect:
            Best_DiceRect=np.mean(mdiceRect)
            Best_epochDiceRect=epochs
        if meanAvePrec >= Best_AvePrec:
            Best_AvePrec=meanAvePrec
            Best_epochAvePrec=epochs
        if np.mean(listIOUNoPP) > bestIOUNoPP:
            bestIOUNoPP=np.mean(listIOUNoPP)
            bestEpochIOUNoPP=epochs
        if np.mean(listIOU) > bestIOU:
            bestIOU=np.mean(listIOU)
            bestEpochIOU=epochs
        
        meanPrec=np.mean(prec)
        meanRec=np.mean(rec)
        meanF1=getF1(meanPrec,meanRec)
        
        # print how many images skipped
        print("skippedImg {} numBlankPred {}/{}".format(
                    skippedImg, 
                    numBlankPred,
                    len(dataset)))
        
        # print best portion
        print("{:2d}: No of test:{}  Dice:{:.6f} (best 102)".format(
                    epochs, 
                    len(mdice[1]), 
                    np.mean(sorted(mdice[1])[100:])))
        print("{:2d}: No of test:{}  Dice:{:.6f} (new) (best 102)".format(
                    epochs, 
                    len(mdice2[1]), 
                    np.mean(sorted(mdice2[1])[100:])))
        print("IOU:{:.6f} Best={} : {:.6f} (before PP)".format(
                    np.mean(sorted(listIOUNoPP)),
                    bestEpochIOUNoPP,
                    bestIOUNoPP))
        print("IOU:{:.6f} Best={} : {:.6f} (after PP)".format(
                    np.mean(sorted(listIOU)),
                    bestEpochIOU,
                    bestIOU))
        print("IOU:{:.6f} (no PP)   (removed worst N)".format(
                    np.mean(sorted(listIOUNoPP)[70:])))
        print("IOU:{:.6f} (after PP)(removed worst N)".format(
                    np.mean(sorted(listIOU)[70:])))

        # get IOU statistics
        analyseIOUStats(listIOU,printStats=False)
        
        # print metrics
        print("{:2d}: No of test:{}  Dice:{:.6f} Best={} : {:.6f}".format(
                    epochs, 
                    len(mdice[1]), 
                    np.mean(avg_dice),
                    Best_epoch, 
                    Best_Dice))
        print("{:2d}: No of test:{}  Dice:{:.6f} Best={} : {:.6f} (new)".format(
                    epochs, 
                    len(mdice2[1]), 
                    np.mean(avg_dice2),
                    Best_epoch2, 
                    Best_Dice2))
        print("{:2d}: No of test:{}  Dice:{:.6f} Best={} : {:.6f} (rect)".format(
                    epochs, 
                    len(mdice2[1]), 
                    np.mean(mdiceRect),
                    Best_epochDiceRect, 
                    Best_DiceRect))
        print("MAP:{:.6f} Best={}:{:.6f}".format(
                    meanAvePrec,
                    Best_epochAvePrec,
                    Best_AvePrec))
        print("IOU:{:.6f} F1:{:.6f} Prec:{:.6f} Recall:{:.6f}".format(
                    np.mean(listIOU),
                    meanF1,
                    meanPrec,
                    meanRec))
        print("AngleError: {:.1f} ({:.1f} Epoch {}) ".format(
                    rmsAngle,
                    bestError['angle'],
                    bestError['epochAngle']))
        print("slenError: {:.1f} ({:.1f} Epoch {})".format(
                    rmsSlen,
                    bestError['slen'],
                    bestError['epochSlen']))
        print("tipError: {:.1f} ({:.1f} Epoch {})".format(
                    rmsTipDistError,
                    bestError['tipDist'],
                    bestError['epochTip'] ))
        print("weightedError: {:.3f} ({:.3f} Epoch {})".format(
                    weightedError,
                    bestError['weightedError'],
                    bestError['epochWeightedError']))
        print("'angle':{:.1f},".format(
                    bestError['angle']))
        print("'slen':{:.1f},".format(
                    bestError['slen']))
        print("'tipDist':{:.1f},".format(
                    bestError['tipDist']))
        print("'weightedError':{:.1f},".format(
                    bestError['weightedError']))
        print("'epochAngle':{},".format(
                    bestError['epochAngle']))
        print("'epochSlen':{},".format(
                    bestError['epochSlen']))
        print("'epochTip':{},".format(
                    bestError['epochTip']))
        print("'epochWeightedError':{}" .format(
                    bestError['epochWeightedError']))

        # get Dice statistics
        analyseDiceStats(mdiceRect,printStats=False)
        
        # display results in table
        netMode=linknetMode
        displayInTable=True
        if displayInTable is True:
            df = pd.read_csv('results.csv', index_col=0)
            df.loc[netMode,'epoch']=bestEpochIOU
            df.loc[netMode,'iou']=bestIOU
            df.loc[netMode,'dice']=Best_DiceRect
            df.loc[netMode,'MAP']=meanAvePrec
            df.loc[netMode,'F1']=meanF1
            df.loc[netMode,'Precision']=meanPrec
            df.loc[netMode,'Recall']=meanRec
            df.loc[netMode,'Angle']=bestError['angle']
            df.loc[netMode,'slen']=bestError['slen']
            df.loc[netMode,'tipDist']=bestError['tipDist']
            df=df.round({
                'iou': 4, 
                'MAP': 4, 
                'F1': 4, 
                'Precision': 4, 
                'Recall': 4, 
                'Angle': 1, 
                'slen': 1, 
                'tipDist': 1})
            print(df)
            df.to_csv('results.csv')
        
        # save as video
        saveVideo=False    
        if saveVideo is True:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_pred = cv2.VideoWriter('videoPred.avi',fourcc, 5, (1000,440))
            
            for x in ppImgCombined:
                #print(x.shape)
                #blank=np.zeros((440,1000,3))
                output_video_pred.write(x)
            output_video_pred.release()
        
        # save images with good dice
        filterImagesBasedOnDice=False
        if filterImagesBasedOnDice is True:
            filteredImDir='filteredImagesIOU'
            if not os.path.exists(filteredImDir):
                os.mkdir(filteredImDir)
                print('created dir:',filteredImDir)
            
            for idx in range(len(filteredImgs['imnames'])):
                # save labels
                savePath=os.path.join(filteredImDir,filteredImgs['imnames'][idx])
                cv2.imwrite(savePath, filteredImgs['masks'][idx])
                
                # save images
                savePath=savePath[:-9]+'.png'
                srcImgDir='/media/mmlab/data/jiayi/data32_washeem_v6/test/'
                srcPath=os.path.join(srcImgDir,filteredImgs['imnames'][idx][:-9]+'.png')
                shutil.copy2(srcPath,savePath)

        # get sample preds
        getSamplePreds=False
        sampleImDir='samplePredsIOU/'+linknetMode
        if not os.path.exists(sampleImDir):
            os.makedirs(sampleImDir)
            print('created dir:',sampleImDir)
        if getSamplePreds is True:
            if linknetMode is 'scse':
                sortIDs=np.argsort(np.array(mdiceRect))
                #sortIDs=np.argsort(np.array(listIOU))
                print('ss',sortIDs[:4],len(sortIDs),type(filteredImgs['imnames']),filteredImgs['imnames'][:4])
                imnamesSorted=np.array(filteredImgs['imnames'])[sortIDs]
                labelsSorted=np.array(filteredImgs['masks'])[sortIDs]
                predsSorted=np.array(filteredImgs['preds'])[sortIDs]
                labelsRectSorted=np.array(filteredImgs['labelRect'])[sortIDs]
                predsRectSorted=np.array(filteredImgs['predRect'])[sortIDs]
                mdiceRectSorted=np.array(mdiceRect)[sortIDs]
                #listIOUSorted=np.array(listIOU)[sortIDs]

                for idx in [0,-30,-29,-28]:
                    print('dice',mdiceRectSorted[idx])
                    #print('IOU',listIOUSorted[idx])
                    
                    # save label
                    savePath=os.path.join(sampleImDir,imnamesSorted[idx])
                    print('savePath',savePath)
                    #cv2.imwrite(savePath,labelsSorted[idx])
                    
                    # save pred
                    savePath=savePath.replace('_mask','_pred')
                    #cv2.imwrite(savePath,predsSorted[idx])
                    
                    # save postprocess pred
                    savePath=savePath.replace('_pred','_predPP')
                    cv2.imwrite(savePath,predsRectSorted[idx])

                    # save postprocess label
                    savePath=savePath.replace('_predPP','_labelPP')
                    cv2.imwrite(savePath,labelsRectSorted[idx])
                    
                    # save original image
                    savePath=savePath.replace('_labelPP','')
                    srcImgDir='/media/mmlab/data/jiayi/data32_washeem_v6/test/'
                    srcPath=os.path.join(srcImgDir,savePath)
                    srcPath=srcPath.replace('samplePreds/scse/','')
                    #shutil.copy2(srcPath,savePath)
                    
            else:
                sampleLabelNames=['p3_11_004_mask.png', # 2nd worst dice 0.1258
                                'p8_17_004_mask.png', # dice 0.7277
                                'p8_17_010_mask.png', # dice 0.7388
                                'p8_19_007_mask.png', # worst dice 0.1074
                                'p8_19_008_mask.png'] # dice 0.7306
                for idx in range(len(filteredImgs['imnames'])):
                    for sampleName in sampleLabelNames:
                        if sampleName == filteredImgs['imnames'][idx]:
                            savePath=os.path.join(sampleImDir,sampleName)
                            #savePath=savePath.replace('_mask','_pred')
                            cv2.imwrite(savePath,filteredImgs['masks'][idx])
                            savePath=savePath.replace('_mask','_pred')
                            cv2.imwrite(savePath,filteredImgs['preds'][idx])
        
        sortIDs=np.argsort(np.array(listIOU))
        sortedNames=np.array(dataset.img_anno_pairs)[sortIDs]
        print('IOU',len(mpath),sortedNames[0],sortedNames[-30],sortedNames[-29],sortedNames[-28])
        
        # print end-of-epoch
        print('-'*80)
        

