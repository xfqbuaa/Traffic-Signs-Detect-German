# German Traffic Signs Detection

## Introduction
This project target is to train a model to detect German 43 classes traffic signs with [GTSDB dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) and [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

Traffic signs detection is the basic of self driving car and should be optimized separately for different countries since there are different traffic signs.

Different from GTSDB competition to detect traffic signs category, this project will detect detail classes for total 43 traffic signs and evaluate model detect performance.  

Two COCO-trained models were used to fine tune, detail you can find [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* ssd_inception_v2_coco
* faster_rcnn_inception_v2_coco

To simplify two trained model with 'ssd' and 'faster' below.

To choose ssd and faster trained model for consideration of accuracy and speed, especially the real time demand of traffic signs detection for self driving car.  

## Performance

### Evaluation metrics
The commonly used mAP metric for evaluating the quality of object detectors, computed according to the protocol of the PASCAL VOC Challenge 2007. The protocol is available [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf).

`metrics_set='pascal_voc_metrics'`
should be added into config file `eval_config`.

### mPA

### Precision recall curves

### Images
* images 1
* images 2
* images 3

## Tensorflow Object Detection installation
This project is based on Tensorflow object detection.

`git clone https://github.com/tensorflow/models.git` is the first step.

[Tensorflow Object Detection API installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

Please compile Protobuf libraries and add Libraries to PYTHONPATH before using Tensorflow object detection API framework.

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
## Data analysis

## Create record

## Training

## SSD vs Faster rcnn

## More

## Resources
[Traffic-Sign Detection and Classification in the Wild](http://cg.cs.tsinghua.edu.cn/traffic-sign/)

[sridhar912 Traffic Sign Detection and Classification](https://github.com/sridhar912/tsr-py-faster-rcnn)

## Licenses
