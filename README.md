#### System Requirements
- RAM : 32GB for tiny-yolo architecture
- Execution time : 5 hours at 8 frame per second

#### Package Requirements
- OpenCV
- Pytorch 0.3.1
- torchvision
- PIL

#### A detailed documentation for each of the components of your system
I have attempted to comment the code out as much possible. It's not possible to copy all of them here.

#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```

#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
Unzip the tar

##### Generate Labels for VOC
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
concatenate 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
wget http://pjreddie.com/media/files/darknet19_448.conv.23
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights
python scripts/voc_eval.py results/comp4_det_test_
```

#### Changes
##### Bounding Box
Normalize the bounding boxes so that bigger and smaller bounding boxes are penalized equally for their localization errors. Also, use smooth L1 loss instead of Mean Squared Error. Smooth L1 loss is less sensitive to outliers and also has been used by (SSD) Single Shot Multibox Detector, Faster R-CNN.

#### Residual Layers
I think we can have more of residual layers than just in the last  layer. They have almost nil computational cost but just have memory costs as the number of feature increases.
Residual layers serve multiple purpose. They allow a network to be deeper because without residual layers a deeper network might not perform as good as its’ counterpart shallow network. They also make the feature vector larger copying coarse features from previous layers through skip connections and identity mapping.
See : cfg/tiny-yolo-shortcut.cfg

#### Mutliscale Training
Randomly selecting the scale of training does not seem good. It is not uniform nor focuses on the scale that the detector is most likely to encounter. It would make more sense if the network was trained on uniform number of images of each scale. 

#### Pooling
It has a lot of max pooling layers. I think we can get rid of it without any performance issue by having strides of 2 in the convolution layers before it. That will also make the network faster because the expensive convolution operations are sparse and we completely got rid of the pooling layers except the last one.
I also think converting the last pooling layer to max pooling would improve the performance because average pooling just averages out the features while a max or min don’t.
See : cfg/tiny-yolo-nopool.cfg

#### More 1*1 filters to reduce dimension
See See : cfg/tiny-yolo-reduce.cfg

#### Focal Loss for classification loss
YOLO9000 uses cross entropy loss for classification errors which penalizes all the errors equally. Focal Loss for Dense Object Detection proposes penalizing the hard cases more i.e. if the classification error is more give it more prominence. The argument is that it allows the network to learn difficult cases faster.

* Cross Entropy (pt) = - log(pt)
* Focal Loss (pt) = -(1 - pt)y log(pt)

Setting y > 0 reduces error when pt  > 0.5 putting more focus on misclassified examples.

#### Optimizers
All the YOLO architectures are trained using Stochastic Gradient Descent optimization algorithm. The authors and other papers as well have not explored other optimization algorithms like RMSProp, AdaGrad and Adam which are shown to converge faster than SGD. This leaves us with the opportunity to use other optimization algorithms to find out algorithms which converge faster for the detection problem but may not have been explored due to the obviously high computational cost of exploring them.
I changed the optimiezr to RMSProp

### Cues
I read most of the referenced papers and more.

#### Assumptions made in the paper. 
* An image contains object(s).
* For classification dataset, they assumed that the predicted bounding box has a 0.3 IOU with the (imaginary) ground-truth. It’s important so that classification dataset can be treated as a detection dataset.
* Correct detection : The bounding box should overlap the groundtruth with Intersection Over Union (IOU) of 0.5 and the class of the object is predicted correctly.

#### Difficulties encountered during the project
* The starting code has zero documentation
* Computationally intensive 
* The project requires pytorch 0.3.1. It’s simply not available on windows officially. It took a lot of time to find it.
* The code originally supported only GPUs. I made it compatible with both GPUs and CPUs by adding CPU variables as well.
* Only the detection code runs properly, the training module of the starting code does not run because at the end of training my test recall, precision and fscores are all 0. It is really hard to say with certainty what’s wrong i.e if the training code / testing code or network architecture is wrong. So I tried to train most of the trainable-light architecture before discovering that the starting source code has some errors. It took a lot of time to discover this as running the training for full 50 epochs takes time.

#### Improvements
None because the code has some errors which causes testing metrics like precision, recall to be 0.

#### Experiemental observations
* 1*1 convolution layers should be used with caution because it may throw away useful features as evident by degradation in detection of larger objects in YOLO3 It's fast though. 
* Residual layers increase the size of feature vector. It could help detection performances' precision and recall metrics.