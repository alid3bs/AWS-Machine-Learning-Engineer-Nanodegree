# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Ali Said Daebis

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
TODO: I noticed that count is continuous not discrete, but at the end the evalutaion matrix don't care about if it is continuos or discrete.
All values were bigger than 0, so it was fine to just submit without doing anything

### What was the top ranked model that performed?
TODO: WeightedEnsemble_L3

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: exploratory analysis find that most of the features are normally distributed, and I extracted hour feature from the datetime feature after converting it to datetype

### How much better did your model preform after adding additional features and why do you think that is?
TODO: the error reduced from 1.7 to 0.65, because this additional feature added alot of information to the model

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: the error reduced from 0.65 to 0.64

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: feature engineering

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
    "model"           : ["initial", "add_features", "hpo"],
    "num_bag_folds"   : [   0,           0,           10  ],
    "num_stack_levels": [   0,           0,           1 ],
    "num_bag_sets"    : [   0,           0,           1  ],
    "score"           : [1.79316,     0.67588,    0.65606]

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/training run.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/kaggle scores.png)

## Summary
TODO: AutoGluon model gives quick intution about the errors and the approch I will take towards the models and data preprocssing
at the end AutoGluon will be a baseline model for another advanced model
