# Underwater Image Classification - Machine Learning ![Project Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBrunosCodeLab%2FUnderwaterImageClasssification-ML&count_bg=%235C9CFF&title_bg=%23008FC9&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)

<div align="center">
    <img src="https://raw.githubusercontent.com/BrunosCodeLab/Images/refs/heads/main/UnderwaterImageClassification-ML/RealizatorBanner.png" alt="Banner" width="1080" />
</div>
<br>

This project was part of a student case contest where our team finished second. Through this experience, we not only focused on developing the best technical solution but also realized that selling the product and presenting it effectively can be just as important—if not more so—than the solution itself. 
The competition allowed us to tackle a real-world challenge, and I am proud of the collaborative effort that made this achievement possible. Working as a team, we combined our strengths and learned valuable lessons about teamwork, innovation, and communication.
<br><br>

## Problem
In the context of underwater infrastructure documentation, the company enables the creation of faithful digital replicas of submerged structures, known as digital twins. 
Modern technology allows us to dive into the depths of the sea and capture high-quality photographs, which are then used to create 3D models. <br> Despite the high success rate of this process, the company faces a challenge where 8% of the collected images are unusable for creating digital twins. 
This large number of irrelevant photos must be manually identified, which results in approximately **25 working days per year spent on reviewing, filtering, and deleting inadequate photos.**

Focusing on optimizing this critical part of the digital twin creation process, this project describes a solution that leverages artificial intelligence to quickly, accurately, and efficiently filter out inadequate photographs by distinguishing good photos from bad ones.
<br><br>

## Solution Overview
The goal of this solution is to develop a deep learning-based system to facilitate the computational analysis and classification of photographs for efficient sorting. 
Using Convolutional Neural Networks (CNNs), we aim to build a model that automatically identifies photo characteristics and categorizes them accordingly. 
This approach has the potential to significantly reduce the time spent on manual sorting, while maintaining a high level of accuracy in classifying the photographs. 
The key part of building this model is training it on a representative dataset (a pre-classified set of photos) and testing it on new, unseen images.
<br><br>

<div align="center">
  <img src="https://raw.githubusercontent.com/BrunosCodeLab/Images/refs/heads/main/UnderwaterImageClassification-ML/Realizator.gif" alt="Gif" width="1080"/>
</div>
<br>

## Solution Development
1. Creating the Dataset for Training the Model <br>
A critical step in developing our automatic photo sorting model is creating a high-quality training dataset. With significant help from Vectrino, we obtained two complete datasets from different projects that had already been classified according to predefined criteria.
With these ready datasets, we can now build and train the model. The first dataset contains 3000 good photos and 3000 bad ones for training, while the second dataset was used for testing the model's reliability.

3. Model Architecture <br>
In this section, we describe the construction of the Convolutional Neural Network (CNN) used to address the task of automatic photo sorting. Using Python code and the Keras library within TensorFlow, we define, train, and evaluate the model.
Models trained on a small number of images (6000 is considered small for neural networks) tend to "overfit," so we implemented restrictions such as "dropout" layers and "patience" during training to prevent this.

5. Training the Model <br>
Here, we delve deeper into the model training process, utilizing advanced techniques such as Learning Rate Scheduling and Early Stopping.
This iterative process aims to adjust the model's weights to minimize the loss function on the training dataset while evaluating the model's ability to generalize on unseen data using a validation set.
<br><br>

## Results and Model Testing
The model was tested on a dataset of 6074 images, 5137 of which were classified as good and 937 as bad. The model sorts incoming images into three classes: Good, Bad, and Review. 
The model classifies any image with over 98% certainty as "Good" or "Bad," while images with less than 98% certainty are sorted into the "Review" category for further inspection by human workers. 
A total of 30 misclassified images were identified within the Good and Bad classes, accounting for 0.49%, while 887 images were sent for review. 
The analysis highlights the need to expand the dataset due to the small number of images with problematic samples (e.g., water surfaces), which creates issues in these 30 cases. 
Increasing the dataset size would enable the model to better generalize and achieve more accurate classification.
<br><br>

<div align="center">
    <img src="https://raw.githubusercontent.com/BrunosCodeLab/Images/refs/heads/main/UnderwaterImageClassification-ML/RealizatorResult.png" alt="Photo" width="500" />
</div>
<br>

## Time Savings: 85% Reduction in Manual Labor
By applying our model to the initial 6074 images, the amount of manual work required was reduced to just classifying 887 images. This represents an 85% reduction in labor. 
If applied to an annual volume of 1,000,000 photos, this optimization would reduce the workload from 25 working days to just 3 working days, leading to significant time savings in the data processing pipeline.
<br><br>
## Conclusion
In conclusion, the implementation of a Convolutional Neural Network (CNN) has proven successful in classifying underwater images, achieving high accuracy on a test dataset of 6074 images. However, some challenges remain, such as misclassifications involving water surfaces, highlighting the need for further enhancement in recognizing specific patterns. 
To improve the model's performance, the company should allow us to expand the dataset, especially with images representing challenging scenarios. 
The addition of better resources, such as more powerful computational capacity, will also help improve the model's complexity and its ability to detect subtle features in the images. 
With these improvements, we could create even more accurate models, leading to better performance in underwater image classification and analysis.


<div align="center">
    <img src="https://raw.githubusercontent.com/BrunosCodeLab/Images/refs/heads/main/UnderwaterImageClassification-ML/Realizator_Placement.png" alt="Photo" width="1080" />
</div>


<h3 align="center">"Alone we can do so little; together we can do so much." – Helen Keller</h3><br>
This project wasn’t just about technology—it was about teamwork. Every line of code, every idea, and every late-night brainstorming session came together because of the amazing people I had the privilege to work with. Thank you all!
