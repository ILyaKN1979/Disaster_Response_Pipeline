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



There are two datasets used to build the model:
-  disaster_categories.csv: contains multi-label categories for each message
-  disaster_messages.csv: contains the actual text messages, both in the original language as well as in English

The data is highly imbalanced.  You can see Figure 1.
<picture>
 <img alt="imbalanced_data" src="https://github.com/ILyaKN1979/Disaster_Response_Pipeline/blob/main/img/imbalanced.png">
</picture>
Figure 1. Imbalanced Data
 
One category (child_alone) does not have a single label =1. 


