# Object detection (dense labeling) & point-based object detector

In global labeling, we label every image with one label.

In semantic labeling, we will have several classes within one image, such as 
humna, animal, sky, background. But we do not distinguish between different animal 
within one image and we label them all as "animal".

In object detection, we are looking for specific object, and similar to semantic labeling,
it requires dense labeling.

Here in this work, the goal is to detect three objects from SuperKartTux computer
game: "karts", "bombs", and "pickups"


The "Object as a Point" technique is used for object detection, which is a more
straight forward method compared to other methods (point-based object detector). The 
detector will represent each object by a single point at its center. This point is 
easy to detect by a dense prediction network (FCN)




## 1 Setting up the FCN model
Many techniques used in object detection and semantic segmentation still applies
for FCN and object detections:

1) Input normalization (in the __init__ section of the model class)
2) Residual blocks in the model
3) Dropout
4) Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
5) Weight regularization
6) Early stopping

As noted above, for the input normalization, we must do it within the __init__ method of our model, so every test
or valid data gets the same input normalization, otherwise it will get messed up:

```python
def __init__(self, layers=[16, 32, 64, 128], n_output_channels=5, kernel_size=3, use_skip=True):
    super().__init__()
    self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
    self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])
```

For FCN model only convolutional operators are used: pad all of them correctly and 
match strided operators with up-convolutions. Skip and residual connections are used.

FCN handles an arbitrary input resolution and produces an output of the same shape as 
the input: using output_padding=1 if needed, or cropping the output if it is too large.
 
The one used here is depicted below:

![insert_pic](pics/FCN_my_note.jpg)


## 2 The output of FCN model
For "Object as a Point" method of object detection, there will be two predicted 
outputs from the FCN:

(a) A heat map: With the same pixed as the input picture

(b) The size of the box around the object

In the models.py, and the Detector class, in the __init__ part:
```python
    self.classifier = torch.nn.Conv2d(c, n_class, 1)
    self.size = torch.nn.Conv2d(c, 2, 1)
```

If we look at the forward method of the Detector class:

```python
return self.classifier(z), self.size(z)

```


## 3 Reduce over-fitting
Similar to a CNN, FCN uses similar techniques to reduce over-fitting issue:

### 3-1 Data Augmentation
The following library can be used to augment the data (training images)
```python
torchvision.transforms.Compose
torchvision.transforms.ColorJitter
torchvision.transforms.RandomHorizontalFlip

```
Debugging the utils.py helps to understand various transformation being done on the
training data.

In the train.py, we get parameters for transformation and augmentation of the 
input data:
```python
parser.add_argument('-t', '--transform',
                    default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
```

The way we get the appropriate classes/object in train.py, by using the user input arguemnt through
the parsing the parameters:
```python
    import inspect
    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})

    train_data = load_data('data/train', transform=transform, num_workers=4)
```
As seen above, when reading the train data, we just need to pass the transfromation we want
into our load_data method.


### 3-2 mess up with the network
```python
torch.nn.Dropout
```


## 4 FCN training
In object-detection FCN training, the DetectionSuperTuxDataset dataset is used. This is
a training dataset that has pixel-level labels for images taken from SuperTux 
computer-game. This dataset accepts a 
data augmentation parameters transform. Most standard data augmentation in torchvision 
do not directly apply to dense labeling tasks. A smaller subset of useful augmentations
are provided that properly work with a pair of image and label in dense_transforms.py.

Since we predict both the heat-map and the size of the box around the box, we will define
two losses for each output. 

```python
det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
size_loss = torch.nn.MSELoss(reduction='none')
```

During the training process in train.py, we need to combine these two losses together and 
come up with one single loss that will be used to update the weight and biases of the FCN
model:
```python
det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
loss_val = det_loss_val + size_loss_val * args.size_weight
```


## 5 Source of semantic data for training
The following shows the location for training data for global labeling and dense
labeling. Here the fous is only to improve the accuracy of global labeling:
```python
!gdown https://drive.google.com/u/3/uc?id=1Gg-SblaraCKqypAKtmrGEO3wgR8uaYaL
!gdown https://drive.google.com/u/3/uc?id=1vwDx1VQeK2GJpSgW7TOulTZgB94AJ85t

!unzip -q supertux_classification_trainval.zip
!unzip -q supertux_segmentation_trainval.zip

!ls
```

## 6 Focal loss
Since the training set has a large class imbalance, it is easy to cheat in a pixel-wise
accuracy metric. 
In utils.py, the distribution of each class in the training dataset mentioned:
```python
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]
```
As seen above, predicting only track and background gives a 96% accuracy. loss function
is defined with the following weight:

```python
w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device)
```

## 7 IOU (intersection over union)
As seen above, loss might not be a good measure in an unbalanced semantic labeling.
Additional measure for  the Intersection-over-Union (IOU) evaluation metric is provided.
This is a standard semantic segmentation metric that penalizes largely imbalanced 
predictions. 

## 8 Some expected results:

Here the confusion matrix is defined for five classes that we are intending doing our
semantic labeling in each picture.
The average IOU (in the diagonal of the confusion matrix), and the subsequent average and global 
accuracy is depicted below:

![insert_pic](pics/iou_measured.JPG)

![insert_pic](pics/avg_accuracy.JPG)

![insert_pic](pics/global_accuracy.JPG)

![insert_pic](pics/loss.JPG)

And for the images, below shows the original image, the labeled image used for traning, and the last one shows
the predicted semantic label for this picture.

![insert_pic](pics/orig_image.JPG)

![insert_pic](pics/labeled_image.JPG)

![insert_pic](pics/predicted_lebel.JPG)



# 8 Reference for Semantic Lableing
https://www.jeremyjordan.me/evaluating-image-segmentation-models/

