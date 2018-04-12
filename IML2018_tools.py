import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Used to save and load models. Speculated to be faster for large models in
#https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/
import joblib
#XGBoost itself
import xgboost as xgb


def loadInputAsDF(fin_name, n = None):
    '''
    Load the dataset from fin_name and transform it into a pd.DataFrame.
    Sample n random entries, if requested (full by default)
    '''
    # This is an ugly workaround to allow reading 
    # of pickle files in pandas 0.19 (Misha's local version), 
    # while the files were produces in pandas 0.21 (on swan)
    if int(pd.__version__.split('.')[1]) < 20:
        import sys
        import pandas.indexes 
        sys.modules['pandas.core.indexes'] = pandas.indexes
        
    df = None
    #read original .npy files
    if '.npy' in fin_name:
        train_array = np.load(fin_name, encoding='bytes')
        train_rec_array = train_array.view(np.recarray)
        df = pd.DataFrame.from_records(train_rec_array)
    elif '.pickle' in fin_name:
        df = pd.read_pickle(fin_name)
    else: 
        print("I do not know how to treat this input file: {}".format(fin_name))
    
    if n :
        df = df.sample(n=n, random_state=314)
        
    df.fillna(-999, inplace=True)
        
    return df
        
        
        
def dropColumns(df, printColumns=True):
    '''
    Drop hard-coded list of columns form the input dataset df.
    '''
    if printColumns:
        print(df.columns)
    
    columns_to_drop = []
    columns_arrays = ['constituents_pt', 'constituents_eta',
       'constituents_phi', 'constituents_charge', 'constituents_dxy',
       'constituents_dz', 'constituents_Eem', 'constituents_Ehad']
    columns_insignificant = ['recojet_eta', 'recojet_phi', 
       'recojet_sd_eta', 'recojet_sd_phi']
    columns_insignificant_const = ['constituents_eta_0',
       'constituents_eta_1', 'constituents_eta_2', 'constituents_eta_3',
       'constituents_eta_4', 'constituents_phi_0', 'constituents_phi_1',
       'constituents_phi_2', 'constituents_phi_3', 'constituents_phi_4',
       'constituents_charge_0', 'constituents_charge_1',
       'constituents_charge_2', 'constituents_charge_3',
       'constituents_charge_4']
    #columns_to_drop.extend(columns_arrays)
    columns_to_drop.extend(columns_insignificant)
    columns_to_drop.extend(columns_insignificant_const)
    #To be done only if those array columns have not been droped yet
    df.drop(columns_to_drop, axis=1, inplace=True)
    if printColumns :
        print(df.columns)
    
    
def evaluate_loss(predictions, truth):  
    #truth is xgb.DMatrix in fact, thust .get_label to get the y column
    if isinstance(truth , xgb.DMatrix):
        t = truth.get_label()
    else:
        t = truth
    ratio = predictions / t
    a = np.nanpercentile(ratio, 84, interpolation='nearest')  
    b = np.nanpercentile(ratio, 16, interpolation='nearest')  
    c = np.nanpercentile(ratio, 50, interpolation='nearest')  
    loss = (a-b)/(2.*c)  
    return loss


def evaluate_loss_xgb(predictions, truth):  
    loss = evaluate_loss(predictions, truth)
    return ('xxx', loss)  


#an adjusted function from this post: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        print("Decided on {} trees".format(cvresult.shape[0]))

    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric=evaluate_loss_xgb)
        
    #Predict training set:
    pred = alg.predict(X_train)
        
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % sqrt(mean_squared_error(y_train, pred)))
    print("Custom loss : %.4g" % evaluate_loss(y_train, pred))
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    
def plotClfPerfEvolution(clf, title='', lossName='xxx'):
    '''
    Do a standardised plot of loss vs boosting iteration
    for the lossName see evaluate_loss_xgb() function
    '''
    x = range(clf.get_params()['n_estimators'])
    evals_result = clf.evals_result()
    
    plt.figure(figsize=(10,6))
    plt.plot(x, 
         evals_result['validation_0'][lossName],
         'b--', label='Train')
    plt.plot(x, 
         evals_result['validation_1'][lossName],
         'r-', label='Test')
    plt.xlabel('N trees')
    plt.ylabel('Desired metrics')
    plt.legend()
    plt.title(title)