# Golf Ball Detection on a Driving Range using Computer Vision

| Student Name       | Student ID | e-mail                             |
| ------------------ | ---------- | ---------------------------------- |
| Christiaan Wiers   |            | ...@student.tudelft.nl             |
| Stijn Lafontaine   | 4908457    | S.C.Lafontaine@student.tudelft.nl  |
| Floris             |            | ...@student.tudelft.nl             |

![Driving_range](/figures/drivingrange.jpg)

## Table of Contents
1. Abstract
2. Introduction
3. Method
4. Results 
5. Conclusion
6. Discussion 

## Abstract
In this project the goal is to research and apply computer vision methods to detect golf balls in a photo of a driving range. This is one of the classic problems in computer vision, namely object detection. The challenge here is that golf balls up front will be larger and more easy to detect for a computer vision algorithm whereas the golf balls more far away and thus smaller are much harder to detect for these algorithms, due to the scarcity of pixel information. The algorithm used to perform the object detection is YoloV4 [1]. This is a state-of-the art object detection algorithm proposed by Bochkovskiy et al. Multiple models were trained using this algorithm. The models differ in datasets and data augmentation used. **ADD RESULTS SUMMARY AND PICTURE RESULTS**


## Introduction
In this project the aim is to detect - as many as possible - golf balls an a driving range. Therefore we research what methods can be used to perform the object detection of golf balls on a driving range. Object detection consists of the localization and classification of multiple objects in the frame of interest. Lots of research has already put into this problem. Another challenge that is specific for our problem is that the objects the algorithm has to detect can be of very different size. Golf balls near the camera are large but those that are far away are then very small. This problem is due to the nature of driving ranges having golf balls at different positions relative to the camera. 

The algorithm we use in this project is YoloV4. We chose this algorithm because it's relatively fast in its prediction and training, has high performance and is easy to use thanks to a lot of community engagement around this algorithm. We also considered other algorithms: RefineDet [4], SNIP [5] and SNIPER [6]. The information to come to these options for our model came from a review of small object detection by Tong et al. [3]. RefineDet is a single shot algorithm, just like Yolo. SNIP and SNIPER are interesting alternatives since these both incorporate scale invariance. Scale invariance object detection could be of use in our project since the objects we try to detect are occurring in different sizes in the same image. The major reason we chose Yolov4 was because that was the only feasible model to train ourselves. RefineDet, SNIP and SNIPER all required hardware and/or packages we did not have or had never worked with before. 

Apart from selecting the right model, gathering data also brings its challenges. We used the internet to gather training data to train our model golf balls. Two datasets were found, so we decided to train on both datasets separately to find differences in quality between the datasets. Since the datasets found on the internet were quite small, data augmentation became necessary to provide the model with the data it needs. The details of this are further explained in the method section. 

At the end, results in testing performance are compared between the different models that were trained on each dataset. 


## Method
In this method section the follow three aspects are described: our chosen network YoloV4, datasets, data augmentation and training method.

### YoloV4
Starting of this project we had to choose a neural network in order to create a detection method that would be most suitable for detecting golf balls on a driving range. During starting phase of this project we have examined the following options for our network: our own implementation of an object region proposal network along with a ResNet-50, RefineDet [4], SNIP [5], SNIPER [6] and YoloV4 [1]. Implementing our own network would not be feasible in our time span. RefineDet, SNIP and SNIPER did either not have code that could be executed by us due to hardware constraints or did not include the necessary documentation to train a model. This was quite disappointing since SNIP and SNIPER seemed very promising due to their scale invariance. This left us with the YoloV4 algorithm. This network had good documentation on how to train a model and community support. Below some more details on this:

1. Easy Google Colab Integration
Training a network as a student is often a challenge since oftentimes you have limited computational power at your disposal. This is where Google Colab comes in handy since they allow users to make use of their GPU's for free (albeit for a limited continuous time of several hours per run of your training). When selecting a network for this project, an easy to integrate network was preferred by us.

2. A lot of community support/engagement
When using neural networks there are a lot of factors that may form timing consuming challenges, examples include:
* The network being build using different -possible more unfamiliar - programming languages such as Caffe or C++.
* Limited reproduction possibilities or explanation on how to train the network, save the weights or what format of labels to use
* Other unforeseen errors

Along with the well acknowledged paper on YoloV4, we also made use of a blog post [7] covering some details about how to train a model yourself.

#### Architecture

##### Yolo history

##### YoloV4
Backbone: CSPDarknet53
• Neck: SPP  PAN 
• Head: YOLOv3 
YOLO v4 uses:
• Bag of Freebies (BoF) for backbone: CutMix and
Mosaic data augmentation, DropBlock regularization,
Class label smoothing
• Bag of Specials (BoS) for backbone: Mish activa-
tion, Cross-stage partial connections (CSP), Multi-
input weighted residual connections (MiWRC)
• Bag of Freebies (BoF) for detector: CIoU-loss,
CmBN, DropBlock regularization, Mosaic data aug-
mentation, Self-Adversarial Training, Eliminate grid
sensitivity, Using multiple anchors for a single ground
truth, Cosine annealing scheduler [52], Optimal hyper-
parameters, Random training shapes
• Bag of Specials (BoS) for detector: Mish activation,
SPP-block, SAM-block, PAN path-aggregation block,
DIoU-NMS

### Datasets
Two different datasets were used to train separate YoloV4 models to compare. The first dataset was found on **WEBSITE DATASET 1*. The second dataset was found on "universe.roboflow.com"

### Data augmentation
We applied data augmentation on both datasets. 

The following techniques were used on the first dataset **NAME**:
1. Brightness increase/decrease
2. Rotate images left/right

The following techniques were used on the second dataset (Roboflow):
1. Brightness increase/decrease

The second dataset did not need the rotation since it contained more datapoints.

YoloV4 itself comes with data augmentation as well. It makes use of the following techniques:
**For the backbone (classifier)[1]:**
1. CutMix
2. Mosaic data augmentation

![CutMix](/figures/cutmix.png)

*Figure 1: CutMix data augmentation example proposed by Yun et al. [8]*

![Mosaic](/figures/Mosaic_data_augmentation.png)

*Figure 1: Mosaic data augmentation example proposed by Bochkovskiy et al. [1]*

**For the detector (region proposal network)[1]:**
1. Self-Adversarial Training
2. Mosaic data augmentation
3. Random data shapes (Resizing inputs before passing through network)


**Self-Adversarial Training** is a form of data augmentation where the forward pass of the network is used to augment the input image. Instead of one forward pass and one backward pass it performs the forward pass twice. First on the input image, then it alters the input image to create the deception that there is no object in the input image. Then it performs forwards pass on that augmented image and then regular backward pass.

Filter out the golf balls with small bounding boxes from the original training dataset and tiling those images (cutting those image up into small smaller images). The ideas behind this is that training on only those tiled and thus very small images will increase the networks ability to detect the small golf balls located at long distances on the driving range since those are small as well.

### Training Method

## Experiments and Results

## Conclusion

## Discussion

## References
1. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.
2. Github yolov4: https://github.com/AlexeyAB/darknet
3. Tong, K., Wu, Y., & Zhou, F. (2020). Recent advances in small object detection based on deep learning: A review. Image and Vision Computing, 97, 103910.
4. Zhang, S., Wen, L., Bian, X., Lei, Z., & Li, S. Z. (2018). Single-shot refinement neural network for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4203-4212).
5. Singh, B., & Davis, L. S. (2018). An analysis of scale invariance in object detection snip. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3578-3587).
6. Singh, B., Najibi, M., & Davis, L. S. (2018). Sniper: Efficient multi-scale training. Advances in neural information processing systems, 31
7. https://techzizou.com/train-a-custom-yolov4-detector-using-google-colab-tutorial-for-beginners/
8. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6023-6032).
