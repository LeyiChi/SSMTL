# SSMTL
SSMTL implements a deep learning-based semisupervised multitask learning model for survival analysis.


## Usage

### Clone the repository:<br />

```
git clone https://github.com/LeyiChi/SSMTL.git
```


### Download the November 2014 update of the National Cancer Institute Surveillance Epidemiology End Results registry (SEER) data:<br/>
1. visit official site [here](https://seer.cancer.gov/data/access.html), submit a request for access to the data, download and put it under ./data.

### Data Preparation
1. Extract data from the downloaded SEER data using sql. The data extraction processes were as follows:
- CRC data
![image](./images/data-extract-crc.png)

convert images and labels from .nii.gz to .npy format
2. set the data path as *data_path*, put images and labels to '*data_path*/images' and '*data_path*/labels', respectively. 
3. run /data_prepare/init_dataset-medical.py

### Requirement
1. PyTorch 1.4.0
2. TensorBoard for PyTorch. [Here](https://github.com/lanpa/tensorboard-pytorch)  to install
3. Some other libraries (find what you miss when running the code)
4. install the GeodistTK [here](https://github.com/taigw/GeodisTK), run
```
    python setup.py build
    python setup.py install 
```
### 5. training
1. coarse-scaled DenseASPP model training:
```
python train_pancreas_c2f200_coarse.py
```
2. fine-scaled DSD-ASPP-Net model training:
```
python train_pancreas_c2f200_saliency.py
```
### 6. testing
1. coarse-scaled model testing:
```
python test_organ_coarse_batch.py
```
2. fine-scaled model testing:
```
python test_organ_fine_batch.py
``` 
