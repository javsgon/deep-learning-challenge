# Deep Learning Challenge: Charity Funding Predictor

## Objective
Create an algorithm using machine learning and neural networks to predict whether applicants will be successful if funded by the nonprofit foundation Alphabet Soup.

## Background
From Alphabet Soup’s business team, there is a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Preprocessing the Data
Using Google Colab, the below pre-processing steps were followed:

1- Read in the charity_data.csv to a Pandas DataFrame, and be sure to 
Identified identified what variable(s) are the target(s) for the model and what variable(s) are the feature(s) for the model.
2- Dropped the EIN and NAME columns.
3- Determined the number of unique values for each column.
4- For columns that have more than 10 unique values, determined the number of data points for each unique value.
5- Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6- Used pd.get_dummies() to encode categorical variables.
7- Split the preprocessed data into a features array, X, and a target array, y. Used these arrays and the train_test_split function to split the data into training and testing datasets.
8- Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Compile, Train, and Evaluate the Model
Used TensorFlow to designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. 

1- Continued using the file in Google Colab.
2- Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3- Created the first hidden layer and choose an appropriate activation function.
4- If necessary, add a second hidden layer with an appropriate activation function.
5- Created an output layer with an appropriate activation function.
6- Checked the structure of the model.
7- Compiled and train the model.
8- Created a callback that saves the model's weights every five epochs.
9- Evaluated the model using the test data to determine the loss and accuracy.
10- Saved and export the results to an HDF5 file. Named the file AlphabetSoupCharity.h5.

## Optimize the Model
Used TensorFlow to optimize the model to try to achieve a target predictive accuracy higher than 75%.

The instructions were to use any or all of the following methods to optimize the model:
- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.
  - Add more neurons to a hidden layer.
  - Add more hidden layers.
  - Use different activation functions for the hidden layers.
  - Add or reduce the number of epochs to the training regimen.

1- Created a new Google Colab file and named it AlphabetSoupCharity_Optimization.ipynb.
2- Imported the dependencies and read in the charity_data.csv to a Pandas DataFrame.
3- Preprocessed again the dataset adjusting any modifications that came out of optimizing the model.
4- Designed a neural network model, adjusting  modifications that  optimized the model to achieve higher than 75% accuracy.
5- Saved and export the results to an HDF5 file. Named the file AlphabetSoupCharity_Optimization.h5.

## Report on the Neural Network Model
Wrote a report on the performance of the deep learning model created for for Alphabet Soup.

The report includes:
1- Overview of the analysis: Explain the purpose of this analysis.
2- Results:
- Data Preprocessing
  - What variable(s) are the target(s) for your model?
  - What variable(s) are the features for your model?
  - What variable(s) should be removed from the input data because they are neither targets nor features?

- Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
  - Were you able to achieve the target model performance?
  - What steps did you take in your attempts to increase model performance?
3- Summary
