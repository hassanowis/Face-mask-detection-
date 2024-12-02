# Face-mask-detection-

## Introduction  
This project focuses on **Face Mask Detection** using deep learning models, leveraging **Transfer Learning** for efficient and accurate classification. The system processes a dataset of approximately **12K images**, categorized into **train, validation, and test** sets.  

Inspired by two significant research works:  
1. *Real-time Detection with Transfer Learning*  
2. *Efficient Mobile-based Detection with Depthwise Separable Convolution (DWS)-Based MobileNet*  

We explored multiple architectures for performance comparison:  
- **MobileNet V2**: Lightweight, efficient, and optimized for mobile deployment.  
- **VGG19**: High accuracy with a more computationally intensive approach.  
- **ResNet50**: Initially used but discarded due to excessive computational cost.  
- **AlexNet**: Moderate performance but suitable for simpler tasks.  

---

## Dataset and Preprocessing  
The dataset comprises **12K labeled images**, already split into training, validation, and test sets.  

### Preprocessing Steps  
1. **Normalization**: Scaling pixel values to [0, 1].  
2. **Balancing**: Ensuring class distribution parity.  
3. **Shuffling**: Randomizing data order.  
4. **Resizing**: Standardizing input image dimensions.  
5. **Data Augmentation**: Using `ImageDataGenerator` to artificially increase data diversity.  

---

## Libraries  
The project uses the following libraries:  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **OpenCV**  
- **TensorFlow/Keras**  

---

## Installation  
To set up the environment, install the required dependencies:  
```bash
pip install numpy pandas matplotlib opencv-python tensorflow termcolor
```

---

## Model Training  
### Steps:  
1. **Data Loading**:  
   Load the dataset using `image_dataset_from_directory` and split into training, validation, and test sets.  

2. **Data Augmentation**:  
   Apply augmentations such as rotation, flipping, and zooming using `ImageDataGenerator`.  

3. **Transfer Learning Models**:  
   - **MobileNet V2**:  
     Lightweight with reduced computation, achieving balanced performance metrics.  
   - **VGG19**:  
     Delivered the best results but is resource-intensive.  
   - **AlexNet**:  
     Lower accuracy, suitable for less complex tasks.  
   - **ResNet50**:  
     Initially used but later discarded due to **high computational time** and inefficiency for real-time applications.  

4. **Model Compilation**:  
   - Loss: `Categorical Crossentropy`  
   - Optimizer: `Adam`  
   - Metrics: Accuracy  

5. **Training and Validation**:  
   Train the models with the training dataset and validate with the validation dataset to monitor performance.  

6. **Testing**:  
   Evaluate the final models on the test dataset to compute the results.  

---

## Evaluation  
### Results Summary  
| Model         | Accuracy | F1 Score | Precision | Remarks                           |
|---------------|----------|----------|-----------|-----------------------------------|
| MobileNet V2  | 93%      | 93%      | 88%       | Efficient for real-time use.      |
| VGG19         | 97%      | 97%      | 99%       | Highest accuracy but resource-intensive. |
| AlexNet       | 84%      | -        | -         | Moderate performance.             |
| ResNet50      | -        | -        | -         | Discarded due to high computational cost. |

---

## Conclusion  
This project demonstrates the trade-offs between accuracy and computational efficiency in face mask detection models. **VGG19** excels in accuracy, while **MobileNet V2** offers the best balance for lightweight and real-time deployment. Models like **ResNet50**, though accurate, were found unsuitable for real-time applications due to excessive computational requirements.  
