### 2018 IML challenge 

The challenge details are decribed in the original wiki: https://gitlab.cern.ch/IML-WG/IML_challenge_2018/wikis/home. 

A small train sample (10k entries) is available in /afs/cern.ch/user/m/mlisovyi/public/iml2018 (CERN AFS cell).
Full samples (~1M entries) are available here: [test](https://cernbox.cern.ch/index.php/s/ODYoAXRfxU6N8U9), 
[train](https://cernbox.cern.ch/index.php/s/EYKKvatjv3XkoR4/download). These are pickled pd.DataFrame objects. The original inputs have been pre-processed by unrolling the 5 hardest contrituents into separate columns and adding sum over all constituents for charge and Eele and Ehad. The original full contituent lists have been droped to save space. 

There are several notebooks provided: 
  * feature extraction (used to create the provided pickle files from the original inputs)
  * XGBoost optimisation (hyperparameter tune)
  * LightGBM test (no thorough optimisation so far). Evaluation metrics on a test sample doesn't work in a LGBM fit for some unclear reason (to be investigated). 
  * evaluation of XGBoost models
  * preparation of submission by predicting on the test dataset

The notebooks are developed in python3.6+, but will most likely work in any python3 environment. 
For some sub-tasks, you will need specific python packages. 
All of those can be installed with conda (either mini- or ana-).
This was tested to work properly on `lxplus` (CERN work servers).
Just do something like
```bash
conda install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```
You might need to add conda-forge to the list of channels: `conda config --add channels conda-forge`,
if you have not done so yet (the installation would fail complaining that a subset of modules can not be found).

Performance of XGBoost was compared to LightGBM with a compatible setup. 
LightGBM was found to perform fits 8x faster, 
while keeping a comparable precision (`1.5`) in terms of the custom loss defined by the challenge.
In particular, XGBoost took `3min 57s` per fit, while lightGBM took only `30.2 s` 
(both averged over 3 execution with 4 parallel jobs).
Even parameter settings is faster in LightGBM: `13.3 µs ± 363 ns` vs `286 µs ± 17.5 µs`
