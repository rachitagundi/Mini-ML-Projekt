# First exercise

## Core terms

- Feature: an input column used to make a prediction.
- Target: the numerical value we want to predict.
- Training data: examples the model learns from.
- Test data: held-out examples used to evaluate the trained model.
- Baseline: a simple comparison. Here it always predicts the training target average.
- Ridge regression: a linear model with a penalty that limits large coefficients.
- MAE: the average absolute difference between actual and predicted targets.
- R²: a score comparing squared prediction errors with a test-mean reference. It can be negative.
- Data leakage: information from outside training data entering the learning process.

## Read the code in this order

1. Find where the dataset is loaded. Inspect data.data.shape and data.data.head().
2. Find the train/test split. Explain why the test rows are held out.
3. Find model.fit and model.predict. Explain how they differ.
4. Find the pipeline. Its scaler learns from training data only.
5. Run the script and compare the two MAE values.
6. Open outputs/prediction_errors.csv and examine its first five rows.

## Questions to answer after running

- What does a prediction error of 40 mean in this dataset?
- Does the trained model beat the baseline?
- Which examples have the largest errors?
- Can this table alone tell us why an error occurred?
- Why does one test split provide limited evidence?
- What would need to change before using a model in a real application?

Write your own observations below after you run and inspect the results.

## My observations

Not recorded yet.
