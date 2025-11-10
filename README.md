# [Enhancing predictive modeling with XGBoost-engineered probabilities and deep neural networks: A hybrid approach for building reliable kinase inhibition QSAR models](https://doi.org/10.1016/j.jmgm.2025.109216)
***DATA &amp; MODELING PROCEDURE.***

## This is a repository to facilitate ChEMBL data curation and hybrid XGBoost/DNN QSAR modeling (classification). Please, if you use our programs or methodology in your research, cite the following paper. https://doi.org/10.1016/j.jmgm.2025.109216.
If you use the programs, make sure you have the requirements listed below and rename your files in the same format of the given examples.
For the math treatment program, first, make sure you have a csv file with molecular descriptors, or use our example : "Descriptors.csv" (DYRK1). Second, save the two-column csv file from the cleaned matrix (containing only the "chembl_id" and the "activity"). Note that in our example (DYRK1) there is also the IC50 column. 


### 1. Requirements
1. Clone the repository : `git clone https://github.com/mohamedmousser/QSAR-MODELING.git`.
2. install python3 : `sudo apt update`,
                     `sudo apt install python3 -y`,
                     `sudo apt install python3-pip -y`.   
3. Install requirements : `pip install requirements.txt`.

### 2. Data treatment
If you wish to build a QSAR model following our pipeline, we provide here a step-by-step tutorial : 
1. Download the csv raw data from https://ebi.ac.uk/chembl in the same directory, or use the IGF1R data example in the repository : "IGF1R.csv".
2. In the linux terminal, type : `python3 ChEMBL_data_preprocessing.py`, you will then be invited to type the name of the raw data file and your new cleaned file will be generated in the same directory.

### 3. Math pre-treatment (MPT)
1. Generate your descriptors file in the same directory, or use the DYRK1 example : "Descriptors.csv".
2. In the linux terminal, type : `python3 MPT.py`, you will be invited to type the name of the descriptors file and your MPT will be done. You will find the correlation matrix and the train/test files in a new directory called "preparation_output".

### 4. Machine learning
In the linux terminal, type : `python3 Machine_learning.py`, you will be invited to type the abbreviation of the kinase and the adequate number of decision trees (these can be found within : [Mousser *et al.* paper](https://doi.org/10.1016/j.jmgm.2025.109216)). A report on the hybrid *XGBoost/DNN* model will appear. 
