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
**LightGBM was found to perform fits 8x faster**, 
while keeping a comparable precision (`1.5` on a local validation sample _without_ pT cuts) 
in terms of the custom loss defined by the challenge.
In particular, XGBoost took `3min 57s` per fit, while lightGBM took only `30.2 s` 
(both averged over 3 execution with 4 parallel jobs on an lxplus host).
Even parameter settings is faster in LightGBM: `13.3 µs ± 363 ns` vs `286 µs ± 17.5 µs`

Performance comparison was also done on a local machine with INTEL i5 2xcore each double-threaded
running Ubuntu 16.4 with a local file access on SSD:
| Number of parallel jobs in training (n_jobs) | 4 | 3 | 2 | 1 |
 ------------------------------
| XGBoost | `5min 11s` | `6min 1s` | `6min 51s` | `11min 52s` | 
| LightGBM | `34.1 s` | `37.9 s` | `40.8 s` | `60 s` | 
Difference in processing speed is even larger (the test was performed averaging over 5 loops of training).

Note, that **scaling with the number of cores goes almost linearly** in processing speed
(there is some initialisation delay at the start of ClassifierSklearnApi.fit(),
which is most likely due to some initialisation, which is potentially not parallel).
However, **scaling into threads on a single core doesn't show a linear scaling**.