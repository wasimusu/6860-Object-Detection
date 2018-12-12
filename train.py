from __future__ import print_function
import sys
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet


# Training settings
datacfg       = sys.argv[1]         # Data configuration
cfgfile       = sys.argv[2]         # Architecture Configuration
weightfile    = sys.argv[3]         # Pretrained weight file

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

# Parse data file and network file
trainlist     = data_options['train']
testlist      = data_options['valid']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = max_batches*batch_size//nsamples+1
use_cuda      = False
seed          = int(time.time())
eps = 1e-2
save_interval = 10  # epoches
dot_interval  = 70  # batches

# Test parameters
conf_thresh   = 0.0  # 0.25
nms_thresh    = 0.4
iou_thresh    = 0.01  # 0.5

# Save training checkpoint in backupdir.
# Create backupdir if it does not exist
if not os.path.exists(backupdir):
    os.mkdir(backupdir)
    
#Fix random number generation
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)  # make model out of cfgfile
region_loss = model.loss        # loss

model.load_weights(weightfile)  # load weights
model.print_network()           # print network

region_loss.seen  = model.seen  # Seen images during training
processed_batches = model.seen//batch_size  # Number of processed batches

init_width        = model.width     # Initial image dimension
init_height       = model.height
init_epoch        = model.seen//nsamples    # How many epochs did we already train

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
# Get the iterable test dataset containing input and targets
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                        shuffle=False,
                        transform=transforms.Compose([
                       transforms.ToTensor(),               # These transformations are executed to every sample
                        ]), train=True),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        # model = model.cuda()  # Original
        model = model.cpu()

params_dict = dict(model.named_parameters())        # All the parameters in the model
params = []
# Changing the hyper-parameters of parameters
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]

# Stochastic Gradient Descent
# optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        # Adjust learning rate according to number of batches trained
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    """ The training module """
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    # load training dataset in the iterator train_loader
    # The transformations are carried out per images
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), 
                       train=True, 
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)     # Adjust learning rate

    logging('epoch %d / %d processed %d samples, lr %f' % (epoch, max_epochs, epoch * len(train_loader.dataset), lr))
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)       # Timing
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        # Optionally change the data for gpu
        if use_cuda:
            data = data.cuda()
            #target= target.cuda()
        t3 = time.time()

        # Make the changes on data and target trackable
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        optimizer.zero_grad()       # set all the gradients of the model to zero | remove previous values
        t5 = time.time()
        output = model(data)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target)  # Compute loss : classification, localization, confidence
        t7 = time.time()
        loss.backward()         # Back-propagate
        t8 = time.time()
        optimizer.step()        # Update the weights of all the parameters
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    # Training parameters
    num_classes = int(cur_model.num_classes)
    anchors     = cur_model.anchors
    num_anchors = int(cur_model.num_anchors)
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data       # Output of the model
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)    # Compute all bounding boxes
        # Find the best boxes
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)      # Non-maximum suppression
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
     
            total = total + num_gts

            # If the confidence of object being in bounding box is greater than threshold count it
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            # Find the box with the highest IOU
            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1     # Index of the best bounding box
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    # Compute precision, recall and fscore
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    # Train for max epochs
    for epoch in range(init_epoch, max_epochs):
        train(epoch)
        test(epoch)
        pass

if __name__ == "__main__":
    pass
