# Managing the end-to-end machine learning lifecycle with MLFlow

This Repository contains information on how to run the MLFlow(nested or otherwise).

# Basic setup

## Setup the environment
- clone this repository
- run the below command in conda environment after changing the root directory to the directory where the repo is cloned
  - pip install virtualenv                     [This installs virtualenv module]
  - virtualenv mlflow_walkthrough              [This creates a new environment called mlflow_walkthrough]
  - mlflow_walkthrough/Scripts/activate        [This activates the environment]
  - pip install -r requirements.txt            [This installs the required libraries]


## Use the below command to start the MLFlow environment. This command creates a directory called mlruns in your cloned repo directory and will contain all the mlflow runs.
- mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000

## Use below link after running the above command to open and manage the MLFlow environment
- http://127.0.0.1:5000

# First go through the walkthrough to understand the MLFlow run.
## The notebook
- run  jupyter notebook
- open hands_on_example.ipynb, then run cell by cell and keep track of it in the MLFlow environment from the link above

### Now instead of going through the notebook one by one, we can a script doing the same thing that notebook does.
- In this case we have `train.py`
- First create an MLproject file which is having information on where the entry point for MLFlow should be when run from command line and what parameters to pass.

### use below command in conda to run the MLFlow using the script train.py
- mlflow run . -P alpha=0.42
- You will be able to see an entry when you refrensh the link and go the the default experiment. as no experiment name was provided while running above command.

## A screeshot of the MLFlow UI has been added to the repo with the below name
- Exercise1_Screenshot_ml_flow_UI

# Now we can run the MLFlow on housing prediction code with data preparation, training and model building
- All three are run as child runs of a master run and can be viewed as such in the MLFlow environment in the link `http://127.0.0.1:5000`

### The MLproject file has already been edited to have two entry points
- main          [This is used when no entry point is mentioned in the command mlflow run]
- secondary     [This is used for the housing prediction]

## Use below command to run the housing prediction python script `exercise2_housing_price_mlflow.py' on the experiment named `Housing_price_prediction'
- mlflow run . -e secondary --experiment-name Housing_price_prediction

### Now You will be able to see your experiment as one master run having three nested child runs in it.
