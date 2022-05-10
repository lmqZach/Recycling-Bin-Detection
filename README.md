# Recycling Bin Detection
## File Structure
```
|-- ROOT
  |-- README.md
  |-- BinDetector_Report.pdf
  |-- Pipfile
  |-- requirements.txt
  |-- run_tests.py
  |-- tests
  |   |-- test_simple.py
  |   |-- testset
  |-- bin_detection
  |   |-- data
  |       |-- ```
  |   |-- roipoly
  |       |-- ```
  |       |-- bin_detector.py
  |       |-- bin_weight.npy
  |       |-- requirements.txt
  |       |-- test_bin_detector.py
  |       |-- test_roipoly.py
```
# Report
## Objective
Train a probabilistic color model to recognize recycling-bin blue color and use it to segment unseen images into blue regions. 

### Detailed Tasks
1. This project utilize the pixel classification method from: https://github.com/lmqZach/Pixel-Classification

2. Given the blue regions, detect blue recycling bins and draw a bounding box around each one. The data and starter code for this part is contained in the folder bin detection. You must implement one of the models from Pixel Classification., either Logistic Regression or Na ̈ıve Bayes or Gaussian Discriminant Analysis for color classification. 

## Overview:
With the increasing trend of building an environment-friendly community, col- lecting recyclables has become very impor- tant. However, finding all blue recycling bins over one road when collecting is still a problem for the human employee. People might not identify each container precisely over a long time. Hence, it will be better if we have an excellent classifier to help hu- man drivers detect the recycle bins and im- prove the efficiency of collecting recyclables.
In this report, I proposed one model that can easily and quickly find and make bounding blue recycling bins. This model first uses one color classifier based on lo- gistic regression, morphological operations, and bounding area extension to identify blue recycling bins.

<img width="406" alt="Screen Shot 2022-05-09 at 16 25 05" src="https://user-images.githubusercontent.com/92130976/167491646-7b3772bc-6979-468a-b34d-b01086603d34.png">

Figure 1: Overview of Training and Testing Process

## Technical Approach
For this problem, other than the 3-color classification in the previous project, we want to generate a binary output to only verify whether this pixel is blue. Hence, our output is a two dimension one-hot encoder, and weight w is an 3 ∗ 2 matrix. Before training, we use roipoly function provided for generate the training labels. Suppose each image has size of (n,m) pixels and 3-RGB channels. The roipoly will generate an (n, m) matrix with each elements in it to be either 0(non blue) or 1(blue).

## Algorithm
We using the following algorithm for training:

<img width="266" alt="Screen Shot 2022-05-09 at 17 21 46" src="https://user-images.githubusercontent.com/92130976/167500686-22c1a025-7503-4871-9a76-fb54f96c3c65.png">

We use the following for algorithm to generate the boundary box:

<img width="263" alt="Screen Shot 2022-05-09 at 17 27 05" src="https://user-images.githubusercontent.com/92130976/167501468-59a1e832-2819-442e-97a0-6ccd20fbd722.png">

When generating the box, we find that, the width of any blue recycle bin is always shorter than length. Hence, we only select the boxes with a greater value in length.

## Result
For Blue Bin Detection, since we only use the train data for training weight w, we do not perform detection during training. For validation, the precision is 100%, and for test, the score is 9.5/10

## Example of Segmented Color Images

<img width="526" alt="Screen Shot 2022-05-09 at 17 35 38" src="https://user-images.githubusercontent.com/92130976/167502574-22c64cf2-af6b-4b79-9965-7dfa51a5f2c1.png">

<img width="526" alt="Screen Shot 2022-05-09 at 17 35 47" src="https://user-images.githubusercontent.com/92130976/167502623-e425424b-a897-467f-af89-7cd633fcfe26.png">

<img width="525" alt="Screen Shot 2022-05-09 at 17 35 57" src="https://user-images.githubusercontent.com/92130976/167502702-ce319b1c-8450-40b9-9478-0c89bfee53c8.png">
