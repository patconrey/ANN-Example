# ANN-Example
This is an example script to create, train, and evaluate an artificial neural network.

# Problem Statement
A bank is losing customers at an alarming rate. They want to investigate why this is happening and identify which
customers are at a higher risk of leaving. If they can identify those customers, then they can further analyze
trends within that customer segment. At the same time, they'll be able to take precautions by reaching out to those
high-risk customers.

# Performance
The trained ANN scores a mean accuracy of 83.8% over 10 folds, each with 100 epochs. The variance of the accuracies 
collected by the 10-fold cross validation was 1.37%.

# Data Set
In an effort to determine the high-risk customers, the bank observed 10,000 customers over six months. They collected
a variety of datapoints they figured would be indicative of retention (or, more technically, "churn"). An excerpt of 
the dataset is provided below

|RowNumber   	| CustomerID  	| Surname  	| CreditScore  	| Geography  	| Gender  	| Age  	| Tenure  	| Balance  	| NumOfProducts  	| HasCrCard  	| IsActiveMember  	| EstimatedSalary  	| Exited  	|
|---	        |---	          |---	      |---	          |---	        |---	      |---	  |---	      |---	      |---	            |---	        |---	              |---	              |---	      |
| 1   	      | 15634602  	  | Hargrave  | 619   	      | France    	| Female  	| 42   	| 2       	| 0       	| 1             	| 1          	| 1               	| 101348.88       	| 1  	      |
| 49   	      | 15766205  	  | Yin       | 550   	      | Germany    	| Male    	| 38   	| 2       	| 103391.38 | 1             	| 0          	| 1               	| 90878.13         	| 0  	      |

# General Architecture
- Input layer with 11 features
- First Hidden Layer with 6 nodes & RELU activation
- Dropout with 10%
- Second Hidden Layer with 6 nodes & RELU activation
- Dropout with 10%
- Output layer with sigmoid activation

- OPTIMIZER: adam
- LOSS FN: binary cross entropy
- BATCH SIZE: 10
- EPOCHS: 100

# Future Work
We want to create an ROC curve for each parameter tuning and use its AUC measure to evaluate the relative performance 
between hyperparameter selections. It would also be interesting to evaluate different architectures, adding another 
hidden layer and increasing the dropout rate of each layer.
