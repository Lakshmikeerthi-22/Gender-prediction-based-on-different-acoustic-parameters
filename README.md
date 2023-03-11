# Classification model to predict the gender (male or female) based on different acoustic parameters.

## Problem statement : 
Create a classification model to predict the gender (male or female) based on different acoustic parameters.

## Context : 
This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded 
voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an 
analyzed frequency range of 0hz-280hz (human vocal range).

## Column Description :
* meanfreq: mean frequency (in kHz)
* sd: standard deviation of frequency
* median: median frequency (in kHz)
* Q25: first quantile (in kHz)
* Q75: third quantile (in kHz)
* IQR: interquantile range (in kHz)
* skew: skewness (see note in specprop description)
* kurt: kurtosis (see note in specprop description)
* sp.ent: spectral entropy
* sfm: spectral flatness
* mode: mode frequency
* centroid: frequency centroid (see specprop)
* peakf: peak frequency (frequency with highest energy)
* meanfun: average of fundamental frequency measured across acoustic signal
* minfun: minimum fundamental frequency measured across acoustic signal
* maxfun: maximum fundamental frequency measured across acoustic signal
* meandom: average of dominant frequency measured across acoustic signal
* mindom: minimum of dominant frequency measured across acoustic signal
* maxdom: maximum of dominant frequency measured across acoustic signal
* dfrange: range of dominant frequency measured across acoustic signal
* modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
* label: male or female

## Dataset :
https://drive.google.com/file/d/1PA_RAM3_c1rd5HY4g9MOoYYys7zlwLn5/view?usp=sharing

## Steps to create the model:

* **Collect the dataset :** We need a dataset with a large number of audio recordings of male and female speakers. The dataset should include different acoustic parameters such as pitch, formants, duration, intensity, and others.

* **Preprocess the data :** We need to preprocess the dataset by removing any missing or irrelevant data. We can also perform feature scaling or normalization to ensure that all input features are on the same scale.

* **Split the data :** Split the dataset into training and testing sets. The training set will be used to train the model, while the testing set will be used to evaluate the model's performance.

* **Train the model :** Use the training data to train a model. This involves fitting the model to the data and adjusting the model's hyperparameters to optimize its performance.

* **Evaluate the model :** Evaluate the model's performance using the testing data. We can use metrics such as accuracy, confusion matrix, classification report, precision, recall, and F1-score to assess the model's performance.

## Result :

![image](https://user-images.githubusercontent.com/94287823/224467878-29eb6101-17ea-4434-b621-71313cd4b0f5.png)
