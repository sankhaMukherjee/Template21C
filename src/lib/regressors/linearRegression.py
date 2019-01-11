import json, os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from joblib   import dump, load
from datetime import datetime as dt

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import cross_validate

def nowF():
    return dt.now().strftime('%Y-%m-%d--%H-%M-%S')

def trainModel(X, y, modelParams, modelFile):

    # Train a linear regressor
    lr     = LinearRegression(**modelParams)
    lr.fit(X, y)
    dump( lr, modelFile )

    scores = cross_validate( lr, X, y, cv = 5, scoring='r2', return_train_score=True )
    scores = pd.DataFrame(scores)
    print(scores)

    # --------------------------
    # Save the trained model
    # --------------------------
    now = nowF()
    folder = f'reports/LinearRegression/trainedModel_{now}'
    os.makedirs( folder )
    
    # ---- Save the model ------
    dump( lr, os.path.join(folder, 'model.joblib') )

    # --- Save the results -----
    scores.to_csv( os.path.join(folder, 'scores.csv'), index=False )

    return

def predict(X, modelFile, y = None):

    lr   = load(modelFile)
    yHat = lr.predict(X)
    
    now    = nowF()
    folder = f'reports/LinearRegression/prediciton_{now}'
    os.makedirs( folder )
    np.save( os.path.join(folder, 'yHat.npy')  , yHat )

    with open(os.path.join( folder, 'modelFile.txt' ), 'w') as f:
        f.write(modelFile)

    if y is not None:
        plt.figure()
        plt.plot(y, yHat, 's', mfc='black', mec='red')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$y$')
        plt.ylabel(r'$y_{Hat}$')
        plt.savefig( os.path.join( folder, 'y-yHat.png' ) )

        plt.figure()
        plt.plot(y, yHat-y, 's', mfc='black', mec='red')
        plt.xscale('log'); #plt.yscale('log')
        plt.axhline(0, color='red')
        plt.xlabel(r'$y$')
        plt.ylabel(r'$y_{Hat}$')
        plt.savefig( os.path.join( folder, 'residuals.png' ) )

        plt.close('all')


    return yHat

def doLinearRegression( modelFile, 
    X, y = None, 
    trainMode = False, testMode = False,
    XHat = None, yHat = None,
    modelParams = None):

    X = np.load( X )
    if y is not None:
        y = np.load(y)

    if trainMode:
        trainModel(X, y, modelParams, modelFile)
    
    if testMode:
        XHat = np.load(XHat)
        yHat = predict(XHat, modelFile, y)


    return
