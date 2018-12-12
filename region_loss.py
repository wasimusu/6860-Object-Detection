import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

use_cuda = False

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = int(num_anchors)                           # Change this to cast to int
    nC = num_classes
    anchor_step = len(anchors)//num_anchors         # Changed this to return int
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][cur_ious>sil_thresh] = 0
    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[int(anchor_step * n)]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = int(num_anchors)
        self.anchor_step = int(len(anchors) // num_anchors)
        # print("Anchor Step type : ", type(self.anchor_step))
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        """
        #output : BxAs*(4+1+num_classes)*H*W
        :param output: contains bounding boxes, confidence for each bounding box being an object and classification score for each box as well
        :param target: contains ground truths
        :return: Nothing
        """
        self.anchor_step = int(self.anchor_step)

        t0 = time.time()
        nB = output.data.size(0)
        nA = int(self.num_anchors)
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        # Making the code both on CPU and GPU
        if use_cuda:
            output = output.view(nB, nA, (5 + nC), nH, nW)     # Just reshaping it
            x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))    # tx
            y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))    # ty
            w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)               # tw
            h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)               # th
            conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)) # confidence of bounding box being an object
            cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1, nC).long().cuda()))             # class score for each bounding box
            cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)       # Reshape / resize
            t1 = time.time()
        else:
            output = output.view(nB, nA, (5 + nC), nH, nW)
            x = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([0]))).view(nB, nA, nH, nW))     # tx
            y = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([1]))).view(nB, nA, nH, nW))     # ty
            w = output.index_select(2, Variable(torch.LongTensor([2]))).view(nB, nA, nH, nW)                # tw
            h = output.index_select(2, Variable(torch.LongTensor([3]))).view(nB, nA, nH, nW)                # th
            conf = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([4]))).view(nB, nA, nH, nW))  # confidence of bounding box being an object
            cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1, nC).long()))                # class score for each bounding box
            cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)   # Reshape / resize
            t1 = time.time()

        # Making the code both on CPU and GPU
        # grid_x, grid_y are offsets cx, cy
        if use_cuda:
            pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
            grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
            grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(
                nB * nA * nH * nW).cuda()

        else:
            pred_boxes = torch.FloatTensor(4, nB * nA * nH * nW)
            grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)
            grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)

            anchor_w = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([0]))
            anchor_h = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1, torch.LongTensor([1]))
            anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)     # pw
            anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)     # ph
            pred_boxes[0] = x.data + grid_x                 # bx
            pred_boxes[1] = y.data + grid_y                 # by
            pred_boxes[2] = torch.exp(w.data) * anchor_w    # bw
            pred_boxes[3] = torch.exp(h.data) * anchor_h    # bh
            pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))       # Keep all the boxes together
            t2 = time.time()

            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                        target.data,
                                                                                                        self.anchors,
                                                                                                        nA, nC, \
                                                                                                        nH, nW,
                                                                                                        self.noobject_scale,
                                                                                                        self.object_scale,                                                                                                        self.thresh,
                                                                                                        self.seen)
            cls_mask = (cls_mask == 1)
            nProposals = int((conf > 0.25).sum().data[0])       # No of proposals with confidence greater than 0.25

        # Making the code both on CPU and GPU
        if use_cuda:
            # Variable wraps a tensor and records the operations applied to it like automatic gradient calculation, etc
            tx = Variable(tx.cuda())
            ty = Variable(ty.cuda())
            tw = Variable(tw.cuda())
            th = Variable(th.cuda())
            tconf = Variable(tconf.cuda())

            coord_mask = Variable(coord_mask.cuda())
            conf_mask = Variable(conf_mask.cuda().sqrt())

        else:
            # Variable wraps a tensor and records the operations applied to it like automatic gradient calculation, etc
            tx = Variable(tx)
            ty = Variable(ty)
            tw = Variable(tw)
            th = Variable(th)
            tconf = Variable(tconf)

            coord_mask = Variable(coord_mask)
            conf_mask = Variable(conf_mask.sqrt())
            cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC))

        tcls = Variable(tcls.view(-1)[cls_mask].long())
        cls = cls[cls_mask].view(-1, nC)        # Actual class prediction output

        t3 = time.time()

        # # Mean Square Error used for the bounding box
        # loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        # loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        # loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        # loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        # loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0

        # Changed that to smooth l1 loss as used by Faster R-CNN and SDD. Is impacted less by outliers
        loss_x = self.coord_scale * nn.SmoothL1Loss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.SmoothL1Loss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.SmoothL1Loss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.SmoothL1Loss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0

        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss
