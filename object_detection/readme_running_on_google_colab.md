# Google Clab

## 1 making sure Cuda GPU is available
```python
import torch

print(torch.cuda.is_available())
```

# 2 Cyncing to the Git Hub repo
```python
import os

os.environ['USER'] = 'hafezbahrami'
os.environ['PASS'] = 'XXXXXX'
os.environ['REPO'] = 'deeplearning'

!git clone https://$USER:$PASS@github.com/$USER/$REPO.git

%cd deeplearning/object_detection
```

# 3 Installing dependencies
Usually all required dependencies are already installed in Google Colab. If not:
```python
!pip install -r requirements.txt
```

# 4 Getting data
The data is either at a URL address, and if zipp'ed we need to unzip it. The following will
get the data intothe current-directory (cd):
```python
!gdown https://drive.google.com/u/3/uc?id=1Gg-SblaraCKqypAKtmrGEO3wgR8uaYaL
!gdown https://drive.google.com/u/3/uc?id=1vwDx1VQeK2GJpSgW7TOulTZgB94AJ85t

!unzip -q supertux_classification_trainval.zip
!unzip -q supertux_segmentation_trainval.zip

!ls
```
The above, will get both general labeling data and dense labeling (segemtnation) data. We only need 
general labeling data here.


The other point is that is the data is on google drive, replace "wget" command with "gdown":
```python
!wget https://www.cs.utexas.edu/~philkr/supertux_classification_trainval.zip
!wget https://www.cs.utexas.edu/~philkr/supertux_segmentation_trainval.zip

```

# 5 Running the tensorboard for observing logs
```python
%load_ext tensorboard
%tensorboard --logdir . --port 6006
```

# 6 run the code
We can specify all the parameters specified in the [if __name__ == '__main__':] part of the train.py.
```python
!python3 -m classifier.train --log_dir .
```

# 6 run the test
We can specify all the parameters specified in the [if __name__ == '__main__':] part of the train.py.
```python
!python3 -m test_classifier classifier -v