# SSMTL
SSMTL implements a deep learning-based semisupervised multitask learning model for survival analysis.


## Usage

### Clone the repository:<br />

```
git clone https://github.com/LeyiChi/SSMTL.git
```


### Download the November 2014 update of the National Cancer Institute Surveillance Epidemiology End Results registry (SEER) data:<br/>
1. visit official site [here](https://seer.cancer.gov/data/access.html), submit a request for access to the data, download and put it under ./data.

### Requirement
1. Keras 2.2.4
2. tensorflow-gpu 2.0.0
3. pycox 0.2.1
4. Some other libraries (find what you miss when running the code)

### Data Preparation
1. Extract data from the downloaded SEER data using sql. The data extraction processes were as follows:
- **CRC data** <br/>
![image](./images/data-extract-crc.png)
- **Lung data** <br/>
![image](./images/data-extract-lung.png)
- **Breast data** <br/>
![image](./images/data-extract-breast.png)
- **Prostate data** <br/>
![image](./images/data-extract-prostate.png)

2. put the extracted data into ./data file directory with the file format .R for R and .csv for python.
3. run python 000-data_process.py to transform categorical variables to one-hot encoded variables.

### training and evaluation for survival analysis with CRs
Go to the survival-analysis-with-CRs directory using 
```
cd survival-analysis-with-CRs/
```

1. Fine-Gray model:
```
Rscript ./001-model-fg.R
```
2. Random survival forest:
```
Rscript ./002-rsf.R
```
3. Simple MLP model:
```
python 003-simpleMLP.py
```
4. DeepHit model:
```
python 004-deephit.py
```
5. SSMTL model:
```
python 005-ssmtl.py
```
6. model performance compare:
```
Rscript ./006-model-compare.R
```
7. variable importance for SSMTL:
```
python 007-ssmtl-vimp.py
Rscript ./008-vimp.R
```
8. nonlinear effects:
```
python 009-ssmtlr-nonlinear.py
Rscript ./010-nonlinear-plot.R
```

### training and evaluation for survival analysis without CRs
Go to the survival-analysis-without-CRs directory using 
```
cd survival-analysis-without-CRs/
```

1. Cox model:
```
Rscript ./001-cox.R
```
2. AFT model:
```
Rscript ./002-aft.R
```
3. Random survival forest:
```
Rscript ./003-rsf.R
```
4. GBM model:
```
Rscript ./004-gbm.R
```
5. DeepSurv model:
```
python 005-deepsurv.py
```
6. DeepHit model:
```
python 006-DeepHit.py
```
7. MTLR model:
```
python 007-MTLR.py
```
8. Logistic Hazard model
```
python 008-logistichazard.py
```
9. PMF model:
```
python 009-PMF.py
```
10. SSMTL model:
```
python 010-ssmtl.py
```
11. model performance compare:
```
Rscript ./011-model-compare.R
```
12. variable importance for SSMTL:
```
python 012-ssmtl-vimp.py
Rscript ./013-vimp.R
```

### results of effects of Prognostic Factors on the Survival Risk were under ./results.



