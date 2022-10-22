# Watch-and-Tell

Did you like to original [Show and Tell](https://arxiv.org/abs/1411.4555) ?

Now it's on video.

![pred](https://user-images.githubusercontent.com/81184255/197354068-834c3258-b953-4cf0-b1c2-9ffa6d26d020.gif)

Here is a mini guide in Jupyter on how to use the Python COCO Api: [PythonAPI.pdf](https://github.com/AndreiMoraru123/Watch-and-Tell/files/9844733/PythonAPI.pdf)

or you can look it up along with the MATLAB and Lua API: https://github.com/cocodataset/cocoapi

### The ___Without Tears___ installation of the COCO dataset

```bash
%%bash

mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/zips/unlabeled2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip unlabeled2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip

cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
unzip image_info_unlabeled2017.zip

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip
```

#### Always do this on Windows

```bash
pip install pycocotools-windows
```

Otherwise nothing will work, like most things on Windows

#### Download the coco weights (I have included the class names and config file here, but these are too big)

```bash
wget https://pjreddie.com/media/files/yolov3.weights
```
