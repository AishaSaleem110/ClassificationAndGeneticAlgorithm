# Classification and Genetic Algorithm

Performing classification using k-nearest neighbour on a real-world dataset and optimising the features using a binary genetic algorithm (GA).

# Background

**Problem 1**
The first problem is a classification problem. The EEG data was collected from 61 active channels (electrodes, see the reference paper for more information but this is not required to complete this submission) where each channel was used to compute a feature. There are 800 patterns from 40 subjects (roughly similar number of alcoholics and controls). The 800 pattern is divided into 3 sets:
1. Training set: 400 patterns (given in train_data.txt file)
2. Validation set: 200 patterns (given in val_data.txt file)
3. Test set: 200 patterns (given in test_data.txt file)

Each row in the file consists of 61 feature values representing either an alcoholic or control subject data. The class labels for the patterns i.e. either alcoholic or control is given as 0 and 1, respectively but only for training and validation data sets. 

**Problem 2**
The second problem is an extension of problem 1. In this problem, we need to use GA to find the appropriate channel (features) to use in the classification that will improve the classification accuracy. There are some additional constraints to be met:
1. Performance (accuracy) ≥ 85%
2. Maximum number of features to use ≤ 40

Both these constraints must be met, otherwise you will get a mark of zero


### The Code

This code is written in Java

### Running The Code
*Open a command prompt window and go to the directory where you saved the java program.
You can run KNN.java or KNN_GA.java or KNN_test.java
Type :
javac KNN.java
Now,  to run your program, type :
java KNN


*The files are self-contained and all necessary libraries are imported.



