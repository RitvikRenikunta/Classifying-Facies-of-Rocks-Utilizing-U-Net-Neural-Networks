# Classifying Facies of Rocks Utilizing U-Net Neural Networks
### Ritvik Renikunta, Joseph Stanley, and Jose Hernandez Meijia
### Department of Petroleum and Geosystems Engineering,
### Department of Computer Science, University of Texas at Austin
# Introduction
This project’s aim is to correctly classify the facies of a rock. Facies are defined as the distinct portions of a rock, which are classified by specific characteristics, such as shape, size, texture, or color [1]. Facies are significant because they play a large part when deciding what part of a rock can yield the most potential fuel [2]. This is necessary when exploring lithologic reservoirs, where rock facies are analyzed to predict the output of oil production from such reservoirs. To improve the current system in which rock facies are analyzed, we can train a U-net, which is a type of convolutional neural network that allows us to conduct image-segmentation.




![image](https://user-images.githubusercontent.com/69605118/169385194-44c1aad0-8d61-473c-b6c4-16a22e949318.png)

Figure 1. The triangular diagram portrays the idea of facies, and how different facies of a singular rock can have different substances and shapes. Each facie is color-coded, depicting the types of facies.

`	`Numerous procedures and methodologies have been used to classify rock facies through image segmentation. Convolutional autoencoder networks such as SegNet, which has minor architectural differences from U-Net, can be used for image segmentation [4]. Specifically, a SegNet differs from a U-Net because, in a SegNet, only max-pooling indices are transferred to the next layer, whereas in a U-Net, the entire feature map is transferred [4]. In addition to this, there are other image segmentation methods such as seed region growing (SRG) which can classify rock facies when combined with spatial information [5]. We decided to use a U-net instead of these options as it is a simple convolutional neural network that is easy to work with.

Our dataset is from “A machine-learning benchmark for facies classification,” authored by Yazeed Alaudah, from the Georgia Institute of Technology [6]. The dataset consists of data of over 1200 different images of rocks with 6 NumPy files. Each file contains many NumPy arrays, which are the numerical representations of the images based on pixel values. Three of these files are the rocks themselves, whereas the other three files are labels of the rocks. These labels are utilized in the training process of the U-net. Each image was represented with the size 701 x 255 pixels.
# Methods
Firstly, image resizing is conducted as the images from the original dataset were 701 x 255 arrays, which are too large for training as it could cause computational overhead. Thus, each image, including the labels, was resized to the size 176 x 64. To make sure no data was lost, a histogram was plotted, including the distribution of every distinct pixel value. The resized image distribution perfectly matched the original image distribution, indicating that approximately no data was lost, as seen in Figure 2. 







![image](https://user-images.githubusercontent.com/69605118/169385234-814105ce-6058-493e-a3ba-28f15342a5a3.png)

Figure 2. The graph above conveys the resized images’ histogram plot overlaid onto the original images’ histogram based on the pixel values.

Next, the U-net architecture was constructed, including multiple layers of downsampling and upsampling. The original architecture is from zhixuhao’s, a user on GitHub, U-Net repository. However, slight modifications from this architecture have been made to accommodate the current training set.






![image](https://user-images.githubusercontent.com/69605118/169385251-f3123bc7-1a8c-4e2b-99c9-c5f5987060b3.png)

Figure 3. The diagram conveys the process of applying a convolutional layer to the inputted training set. After the convolution, max-pooling is applied to reduce the dimensions of the image [3].

As seen in Figure 3, a convolutional layer is applied twice, which is then followed by a max-pooling layer. This process occurs four times, and in each step, the dimension of the image is further reduced. Furthermore, in this process, two dropout layers are placed to reduce the model’s complexity. 

![image](https://user-images.githubusercontent.com/69605118/169385279-10f2b653-8fe4-458a-84e9-742570b44c9a.png)

Figure 4. In this image, two convolutional layers are applied with a filter size of 3x3. After this, an upsampling-2D layer is utilized to increase the size of the image [3].

As seen in Figure 4, there are two convolutional layers along with an upsampling layer that follows immediately. This process is conducted after the downsampling, as shown in Figure 3. The current process is applied four times. Finally, after the downsampling and upsampling, the outputted image is the same as the size of the inputted image: 176 x 64. This architecture should output an image with predicted labels, classifying the distinct facies of the inputted rock image. After training the model, hyperparameter tuning was conducted in order to potentially increase the accuracy of the model. The two hyperparameters that may have affected the model’s results are the dropout rate, which is used for the dropout layer in the convolutional network, and the learning rate, which is used in the Adam optimization function. In order to optimize these hyperparameters, both Bayesian and Hyperband optimizations were conducted.
# Results
The U-Net model was trained for 20 epochs, with a batch size of 32 and a learning rate of 0.0004. Moreover, in order to detect overfitting, a validation split of 0.1 was also included. After training, the model reached an accuracy of 96.31% and a loss of 9.37%. 

![image](https://user-images.githubusercontent.com/69605118/169385296-82a71f56-5359-4b5a-a688-7b36b4500815.png)

Figure 5. The graph above conveys the accuracy percentage during training, as conveyed by the epochs. The validation set’s accuracy is depicted by the orange curve, whereas the training set’s accuracy is depicted by the blue curve.

![image](https://user-images.githubusercontent.com/69605118/169385306-5bcf05e3-7816-460a-877e-90235a51e710.png)

Figure 6. The graph above portrays the loss as the epochs progressed during training. The validation set’s loss is depicted by the orange curve, whereas the training set’s loss is depicted by the blue curve.




![image](https://user-images.githubusercontent.com/69605118/169385331-8b302434-cbe8-4407-bf83-f781760b9423.png)

Figure 7. The image shows a side-by-side comparison of the predicted labels from the model (the left image) and the true labels (the right image). This was taken from the from the training set.

![image](https://user-images.githubusercontent.com/69605118/169385519-631b0242-ad0a-4907-9838-d22e644c2e39.png)

Figure 8.The image shows a side-by-side comparison of the predicted labels from the model (the left image) and the true labels (the right image). This was taken from the testing set.

Next, for hyperparameter tuning, the Bayesian optimization model ran for 5 trials, for 15 epochs each, along with a 0.1 validation split. Additionally, the Hyperband optimization model ran for 16 trials, each with 15 epochs, until its calculations converged. The Bayesian model outputted an optimal learning rate of 1e-5 with a dropout rate of 0.5, whereas the Hyperband model suggested an optimal learning rate of 1e-4 with a dropout rate of 0.8. 

Utilizing the Bayesian-optimized hyperparameters, the model then was re-trained for 20 epochs. The model received a 92.11% accuracy and a 20.05% loss.

![image](https://user-images.githubusercontent.com/69605118/169385546-4e13ebed-3213-4d4e-a4a5-bb56aa748b49.png)

Figure 9. The image above conveys both the accuracy and loss of the model, utilizing the Bayesian-predicted hyperparameters, over time. The blue curve depicts the training set’s accuracy/loss, whereas the orange curve depicts the validation set’s accuracy/loss.

![image](https://user-images.githubusercontent.com/69605118/169385578-bccf370d-e504-43bf-b04e-d3136ce1daa2.png)

Figure 10. The image above conveys a side-by-side comparison of the Bayesian model’s predicted labels, on the left, and the actual labels, on the right. This was taken from the testing set.


Utilizing the Hyperband-optimized hyperparameters, the model was once again re-trained for 20 epochs. The model received an accuracy of 94.11% and 46.91% loss.

![image](https://user-images.githubusercontent.com/69605118/169385607-95e540f8-9498-4028-886a-4b6a42ed059f.png)

Figure 11. The image above conveys both the accuracy and loss of the model, utilizing the Hyperband-predicted hyperparameters, over time. The blue curve depicts the training set’s accuracy/loss, whereas the orange curve depicts the validation set’s accuracy/loss.

![image](https://user-images.githubusercontent.com/69605118/169385632-734d64ee-d9de-43b7-8a07-18266ffc43f2.png)

Figure 12. The image above conveys a side-by-side comparison of the Hyperband model’s predicted labels, on the left, and the actual labels, on the right. This was taken from the testing set.
# Discussion
As seen in Figure 5, the accuracy increased over time, and it eventually plateaued. Portrayed by Figure 6, the loss decreased over time, and likewise to the accuracy, it also eventually plateaued. The validation set’s curve closely matched the training set’s curve, which implies that overfitting is not taking place. Looking at Figure 7 and Figure 8, the expected and predicted labels are very similar, implying that the model is accurate for the training set, even with decreased dimensionality. However, looking at the prediction for a rock in the testing set, the labels don’t match as well as they did with the training set. To potentially fix this, hyperparameter tuning was conducted to reduce the model’s loss for the testing set’s predictions without compromising the accuracy of the training set’s predictions. 

Firstly, utilizing the returned values from Bayesian optimization, which were 1e-5 for the learning rate and 0.5 for the dropout rate, the model outputted a 92.11% accuracy. Looking at Figure 9, it is evident that there is not much overfitting as the validation set’s accuracy/loss aligns closely with the training set’s accuracy/loss. However, as seen in Figure 10, when attempting to make a prediction on the testing set, an entire label is ignored almost completely, resulting in unsatisfactory results.

Next, utilizing the returned values from Hyperband optimization, which were 1e-4 for the learning rate and 0.8 for the dropout rate, the model outputted a 94.11% accuracy. While it may seem that this model would be more accurate at predicting testing labels than the Bayesian model, in fact, the opposite is true. Looking at Figure 11, there is a clear difference between the training set’s accuracy/loss and the validation set’s accuracy/loss. This implies overfitting, resulting in poor testing set predictions. This is evidenced by Figure 12, where the predicted labels largely differed from the actual labels.
# Conclusion
Overall, while the models have high training set prediction accuracy, as they are above 90%, the training set prediction accuracy is evidently lower. As seen in Figures 8, 10, and 12, the original model’s testing set prediction accuracy is the highest, resulting in the most acceptable results. This implies that the hyperparameter tuning may not have worked as it should have. To address this issue, possibly more epochs are required along with more trials. Moreover, to decrease the model’s loss, it may be beneficial to add batch size as a hyperparameter to optimize through Bayesian and Hyperband techniques. Furthermore, additionally to this U-Net, additional architectures could be created to possibly enhance facies label prediction accuracy. This includes ResNet, HRNet, or SegNet architectures, as discussed earlier in this paper. After training the data with these new architectures, the results, including accuracy and loss, must be compared to the U-Net, which will convey any possible significant differences between the models.
# References
[1] *Facies*. Historical Geology. (n.d.). Retrieved March 28, 2022, from https://opengeology.org/historicalgeology/facies/ 

[2] AI, X., & Wang, H. (2019). Automatic Identification of Sedimentary Facies Based on a Support Vector Machine in the Aryskum Graben, Kazakhstan. 

[3] *NN-svg*. NN SVG. (n.d.). Retrieved March 28, 2022, from http://alexlenail.me/NN-SVG/LeNet.html 

[4] Karimpouli, S., &amp; Tahmasebi, P. (2019). Segmentation of digital rock images using deep convolutional autoencoder Networks. Computers & Geosciences, 126, 142–150.

[5] Liu, J., Dai, X., Gan, L., Liu, L., &amp; Lu, W. (2018). Supervised seismic facies analysis based on image segmentation. GEOPHYSICS, 83(2).

[6] Alaudah, Yazeed, Michalowicz, Patrycja, Alfarraj, Motaz, & AlRegib, Ghassan. (2019). Facies Classification Benchmark (1.0) [Data set]. Zenodo.

