## Cloth Segmentation

This git includes implementations of object detection, semantic segmentation and gan(pixel2pixel). We mainly do this for cloth segmentation.

<p>
<img src="https://github.com/allyLei/deepvision/blob/cloth/data/images/0240.jpg" width="200" height="320" />
<img src="https://github.com/allyLei/deepvision/blob/cloth/data/images/0561.jpg" width="200" height="320" />
<img src="https://github.com/allyLei/deepvision/blob/cloth/data/images/0579.jpg" width="200" height="320" />
<img src="https://github.com/allyLei/deepvision/blob/cloth/data/images/0687.jpg" width="200" height="320" />
</p>

### Installation on Linux

1. Install python3 and [pytorch](http://pytorch.org/), note that this program only supports python3, choose a pytorch version that matches with your cuda, for a python3 environment, we recommend install [anaconda3](https://www.anaconda.com/), since it also installs other useful tools includes numpy.

2. Run other installation: `./install.sh`, this includes some dependencies, model and data downloading, etc


### How to train a cloth segmentation model

After installation, rain a cloth segmentation model is easy, just to run

```
./run.sh
```

If it does not run successfully, please check first whether you have installed cuda and compatible pytorch version, then check whether you use python3 environment, 


Some hyper-parameters can be set in the run.sh file, including

    CUDA_VISIBLE_DEVICES: one single card (1) or multiple cards (0,1,2,3)
    --num_classes: default 13 
    --data_dir: training data directory, default data/clothing 
    --ckpt_dir: the trained model checkpoint directory, default ./ckpt
    --pretrained: pretrained model checkpoint path, default: data/pretrained_ckpt/resnext101_32.pth 
    --batch_size: batch_size, default is 16 
    --optim: optimizer selection, including adam, sgd, default is adam
    --lr_policy: learning rate decay policy, default poly 
    --lr_power: only valid when your lr_policy is poly, indicate the poly decay power, default 0.9
    --max_epochs: training total epoches, default 200
    --weight_decay: weight decay, default 0.0001 
    --lr: base learning ratio, default 0.0002
    --summary_step: how many steps to visualize our training loss/seg result/data transformation, default 50 
    --save_step: how many steps to save one model checkpoint file, default 200


Some important hyper-paramters are `lr`, `max_epochs`, `batch_size`, for others we recommend to use the default. `lr` can be set within a range of `[0.00005, 0.0008]`, `max_epochs` indicates the total number of epochs to train, normally, it shouldn't be this large, but in our case, since the training number is a few thousand, you can set between 50 to 200, the value of `batch_size` should fit your gpu memory, if you have error message like out of memory, then you should reduce this value
or use multiple cards (when using multiple cards, ensure this batch_size should be divisible by number of gpu cards). From our experiments, we find that large batch size hurts the final performance, consequently, batch size should be in 12 to 32. 


### Training stage visualization

While you are training, you can moniter the training process through your browser. Open `http://ip:8097`, you can see the loss decay curve along with the learning rate decay curve, during each `save_step`, we save the current model checkpoint file, and do evaluation, and visualization the current model segmentation result.

<img src="https://github.com/allyLei/deepvision/blob/cloth/data/images/visdom.png" width="800" height="600" align=center />

### Clothing dataset

Clothing segmentation dataset includes 3548 training images and 94 evaluation images. All images can be downloaded from `git@git.liebaopay.com:cmvideo/model-data.git`. In this dataset, there are total 13 categories: background, skin, hair, outer top, inner top, skirt, dress, outer bottom, inner bottom, shoes, bag, accessories and other. You can check [this file](http://git.liebaopay.com/sunlei/deepvision/uploads/433bb23844815fa225e1ba174eb03117/%E6%9C%8D%E8%A3%85%E5%9B%BE%E7%89%87%E6%A0%87%E6%B3%A8%E6%A0%87%E7%AD%BE%E8%AF%B4%E6%98%8E-13%E7%B1%BB%E5%88%AB.pdf) for more details about the definitions of each categories.

We evaluate mIoU metric of the 94 evaluation images. Our model achieves `0.697` mIoU. We also provide IoU of each category:

```
 {'IoU': 0.98622455047595303, 'class': u'background'},
 {'IoU': 0.84063120522432955, 'class': u'outer top'},
 {'IoU': 0.81003161089349907, 'class': u'skin'},
 {'IoU': 0.79312295157607704, 'class': u'outer bottom'},
 {'IoU': 0.78137921405634136, 'class': u'shoes'},
 {'IoU': 0.76459341979188555, 'class': u'hair'},
 {'IoU': 0.7287554889432305, 'class': u'bag'},
 {'IoU': 0.66997200282999914, 'class': u'skirt'},
 {'IoU': 0.63779524213652439, 'class': u'dress'},
 {'IoU': 0.62010812092940515, 'class': u'inner top'},
 {'IoU': 0.58730778312684029, 'class': u'inner bottom'},
 {'IoU': 0.43811378494922798, 'class': u'other'},
 {'IoU': 0.40354276517798532, 'class': u'accessories'}
```

### How to use a trained checkpoint model

Copy your trained checkpoint model file `model.pth-xxx` into `demo/server/ckpt/cloth/`, you can replace the `ckpt_path` by your own checkpoint file in the `demo/server/conf/cloth_320.conf` file. Our demo includes a server part and a client part. You can call the server and get the segmentation result, meanwhile, you can also directly see the performance by installing android app.

--------

### Problems?

If it does not run successfully, please check one by one:

    1. whether you have installed cuda and compatible pytorch version

    2. check whether cuda path is added into your .bashrc
    
    3. check whether you use python3 environment 

    4. check the training data have been downloaded (it should be in the 'data->clothing' directory)

    5. check the pretrained ckpt files have been downloaded (in the directory of 'data->pretrained_ckpt')

You can add cuda path into your .bashrc using following:

```
export CUDA_HOME=/usr/local/cuda/
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:$CPLUS_INCLUDE_PATH
```



