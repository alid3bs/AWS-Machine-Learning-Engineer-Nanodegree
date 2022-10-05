# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dog vs cats classification dataset which can be found in the https://www.kaggle.com/datasets/tongpython/cat-and-dog?select=test_set.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
![data](https://user-images.githubusercontent.com/70338979/193952779-0af1785a-c052-4298-930d-d65e10b17138.PNG)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
-I used Resnet18, because its powerfull and has a little weights compared with other pretrained models.
-Parameters used => "lr", "batch_size", "epochs" with ranges => (0.001, 0.01), ([16,32,128]), (3,6), respectively.

Remember that your README should:
- Include a screenshot of completed training jobs
 ![training_jobs](https://user-images.githubusercontent.com/70338979/193952681-4ddb7497-6945-4e19-a1d9-1bf955d9ff80.PNG)

- Logs metrics during the training process
objective_metric_name = "average test accuracy"
objective_type = "Maximize"
metric_definitions = [{"Name": "average test accuracy", "Regex": "Testing Loss: ([0-9\\.]+)"}]

- Tune at least two hyperparameters
![best hypers](https://user-images.githubusercontent.com/70338979/193952949-0e97e6aa-bc63-4bef-af30-2b2fe51d9066.PNG)

- Retrieve the best best hyperparameters from all your training jobs
![best hypers](https://user-images.githubusercontent.com/70338979/193952739-d17ea21a-6637-4700-8c6c-c2e606c8eb53.PNG)

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![endpoint](https://user-images.githubusercontent.com/70338979/193952850-b6ca5eea-3512-42ad-b909-e5e05dc4d788.PNG)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
