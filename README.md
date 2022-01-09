# Skin-Cancer-Detection

## Skin Cancer MNIST:HAM10000 Dataset

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

![1_XbDGv1EBthwcnaCz-yp-9A](https://user-images.githubusercontent.com/50828923/148700267-6a94f2ca-d914-439d-bf11-f0843cb4d3cc.png)


**Data Pre-Processing**

The ResNet50 and MobileNetV2 transfer learning models were applied to the Skin Cancer MNIST:HAM10000 dataset (‘the dataset’) using PyTorch. The dataset was split into training, validation, and test sets at a ratio of 80/10/10 of total observations. All images were resized to 224x224 pixels and normalized by the mean and standard deviation of the three colour channel of the dataset. Observations with multiple class labels were removed from the validation and test sets. Training observations were multiplied in order to deal with class imbalances and augmented to include RandomVerticalFlip, RandomRotation and ColorJitter transformations.

**Modelling Parameters**

When applying both of the transfer learning models and for the purposes of com- parative analysis, the Adam optimizer with a learning rate of 0.001, Cross Entropy Loss criterion and batch size of 32 was trained for 10 epochs. The resulting model after each training epoch was passed to the validation set. Both models were trained and run using a NVIDIA Tesla P100 GPU. The results of each of these models are discussed below.

**Model 1: ResNet50**

The ResNet50 model consists of 48 convolutional layers, 1 MaxPool layer and 1 Average Pool layer. This model architecture has demonstrated successful performance when applied to computer vision tasks such as image classification [1]. The ResNet50 model is a deep residual network which was pre-trained on ImageNet [2]. The resulting training performance and classification performance are shown below.

<img width="371" alt="Fig 1" src="https://user-images.githubusercontent.com/50828923/148700169-301a98b7-c0ad-4e63-afec-943067d3f7c1.png">

<img width="352" alt="Screen Shot 2022-01-10 at 7 52 18 am" src="https://user-images.githubusercontent.com/50828923/148700558-502ca684-010e-4612-97ec-a9db32d86a22.png">



**Model 2: MobileNet V2**

The MobileNet V2 model is a convolutional neural network architecture that is opti- mized for performance on mobile devices. The network architecture is optimised for speed and memory; and contains residual blocks [3]. The resulting training perfor- mance and classification performance are shown below.

<img width="372" alt="Fig 6" src="https://user-images.githubusercontent.com/50828923/148700222-590c53a1-bf97-4871-8186-775da79f6810.png">

<img width="353" alt="Screen Shot 2022-01-10 at 7 51 55 am" src="https://user-images.githubusercontent.com/50828923/148700568-0b3f4af1-2e1b-449c-b785-bb6ebf303398.png">


**Comparing Model Performances**

The training times for both models were similar with the ResNet50 model training for a total time of 1:26:41 and the MobileNetV2 model training for a total time of 1:23:37. Both models demonstrated in similar accuracy and loss curves during training.
Given that a significant class imbalance exists in the original dataset and the test dataset, the weighted average f1-score was used as the most appropriate metric for evaluating the performance between the two models [4]. A comparison of the weighted average f1-scores between the two models during testing shown by the respective classification reports which indicate that both models have demonstrated equally successful performances in this classification task with both models achieving ~90% classification accuracy.

**Conclusion**

The ResNet50 and MobileNetV2 transfer learning models were trained on the MNIST:HAM1000 dataset following pre-processing of the dataset. The predictive performance of these models was then evaluated using a separate test set. Both models have demonstrated equally successful performances in this classification task with both models achieving ~90% classification accuracy.

**References**

[1] He, K. et al.: Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770-778 (2016).

[2] PyTorch, https://pytorch.org/hub/pytorch_vision_resnet/.

[3] PyTorch, https://pytorch.org/hub/pytorch_vision_mobilenet_v2/.

[4] Van Rijsbergen, C.: Information Retrieval (2nd ed.). Butterworth-Heinemann (1979).
