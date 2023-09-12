# Car-vs-Bike-Image-Classification
<div class="sc-emEvRq gZqHzs sc-hGFITe jXToHY"><p>The "Car vs Bike - Image Classification" dataset is a collection of images downloaded from various sources, containing photographs of both cars and bikes. This dataset has been compiled for the purpose of training and evaluating image classification algorithms.</p>
<p>The dataset contains a total of 4000 images. The images have been labeled as either "Car" or "Bike" and are stored in separate directories.</p>
<p>This dataset can be used for a variety of tasks related to image classification, including developing and testing deep learning algorithms, evaluating the effectiveness of different image features and classification techniques, and comparing the performance of different models.</p>
<p>Researchers and practitioners interested in using this dataset are encouraged to cite the original sources of the images and to acknowledge any modifications made to the dataset for their particular use. The dataset may be useful for tasks such as automated bikes sorting or road worthiness analysis.</p>
<p>This data set is a collection of 2,000 Bike and Car images. While collecting these images, it was made sure that all types of bikes and cars are included in the image collection. This is because of the high Intra-variety of cars and bikes. That is, there are different types of cars and bikes, which make it a little tough task for the model because the model will also have to understand the high variety of bikes and cars. But if your model is able to understand the basic structure of a car and a bike, it will be able to distinguish between both classes.</p>

<p>The data is not preprocessed. This is done intentionally so that one can apply the augmentations you want to use. Almost all the 2000 images are unique. So after applying some data augmentation, you can increase the size of the data set.

The data is not distributed into training and validation subsets. But one can easily do so by using an Image data generator from Keras.</p>
<h3>Main Objective of the analysis</h3>
<p>In this report, I will be using deep learning models in classifying 2 distinct images 'Cars' and 'Bikes' gotten locally in my computer. Those images were separated into training and validation directories initially defined, where I performed data augmentation with image data generators. I used CNN based model for training the images and also pre-trained models such as InceptionV3 and VGG16. In the process of training, we set the layers of the pre-trained model as non-trainable (frozen) to retain their learned features while adding new layers for fine- tuning the model for binary classification.</p>

<h3>Deep Learning Analysis and Findings</h3>
<p>After conducting experiments with three different models, we observed that the Inception-based model consistently outperformed the other two. Specifically, VGG16 exhibited a lightweight nature and faster training compared to the InceptionV3 model. Interestingly, InceptionV3 demonstrated quicker convergence than VGG16 during training. However, our plain custom model failed to converge effectively. In light of these findings, our final recommendation is to utilize the InceptionV3 Model due to its superior performance and relatively efficient convergence.</p>

<h3>Model Flaws and Strength and findings</h3>
<p>In our training process, we employed a limited number of training documents and relied on image augmentation techniques. This approach raises the concern of potential overfitting, where the model may become too specialized to the training data. To address this issue effectively, increasing the sample size for training data could be a beneficial step, providing a more diverse and representative dataset.</p>

<h3>Advanced Steps</h3>
<p>Additionally, aside from considering the Inception model, we might explore alternative architectures such as YOLO or ResNet to enhance model convergence. These models could offer improved performance and convergence characteristics, potentially further enhancing our model's ability to generalize to new data.</p>

For more insights on this project "Car vs Bike Image Classification," please check the presentation <a href='https://github.com/Izuchukwu234/Car-vs-Bike-Image-Classification/blob/main/Analysis%20on%20Car%20vs%20Bike%20Image%20Classification.pdf'>here</a>
</div>
