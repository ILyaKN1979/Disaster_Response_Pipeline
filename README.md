# The project - "Disaster Response Classification Application"


### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)

## Installation <a name="installation"></a>

There should be a distribution of Python versions 3.11*.

- Clone this repository
- Install required packages by run 'pip install -r requirements.txt'
- Choose the correct folder (cd Disaster_Response_Pipeline)
- Activate environment:  
		python3 -m venv venv
		.\venv\Scripts\activate  
		
- To run ETL pipeline that cleans data and stores in database
	run: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves
	run: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- cd into the app directory and run 'python run.py'
- open http://127.0.0.1:3000 on your browser to view the web app


## Project Motivation<a name="motivation"></a>
Super project from Udacity - Disaster Response Classification 

1) Project Workspace - ETL:

The initial stage of your data pipeline involves the Extract, Transform, and Load (ETL) process. In this step, you will retrieve the dataset, perform data cleaning using pandas, and subsequently save it into an SQLite database.

Prior to this, exploratory data analysis was conducted to determine the desired approach for preparing the dataset.

There are two datasets used to build the model:
-  disaster_categories.csv: contains multi-label categories for each message
-  disaster_messages.csv: contains the actual text messages, both in the original language as well as in English

The data is highly imbalanced.  You can see Figure 1.
<picture>
 <img alt="imbalanced_data" src="https://github.com/ILyaKN1979/Disaster_Response_Pipeline/blob/main/img/imbalanced.png">
</picture>
Figure 1. Imbalanced Data
 
One category (child_alone) does not have a single label =1. 
####Therefore, in order not to synthesize the data, I violated the task a little and did it for 35 categories for which there is data!!!

2) Project Workspace - Machine Learning Pipeline
For the machine learning portion, the data was split into a training set and a test set. Then, a machine learning pipeline was created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 35 categories!!! (multi-output classification). Finally,  model would be saved to a pickle file. 
I made a lot of investigation, tried to use different model as Random Forest, Logistic Regression, and Decision Tree, with different parameters for Grid Search. 

The Best results were for Logistic Regression: 
('clf', MultiOutputClassifier(LogisticRegression(class_weight="balanced",
                                                         random_state=88, C=0.1,
                                                         max_iter=100,
                                                         solver='liblinear')))

So this model was chosen as the base model. To simplify fitting the model for Grid search next parameter was used: 
parameters = {
        'clf__estimator__penalty': ['l1', 'l2'],  
                }
We only choose  the  type of regularization (L1 or L2)

3) Web app - In the last step, a Flask web app displays the results of the model's work. 
