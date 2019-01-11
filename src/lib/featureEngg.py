import json, os
import pandas as pd 
import numpy as np

from sklearn import preprocessing
from joblib import dump, load

def doFeatureEngineering(rawData, rawDataFormat, trainingMode,
    dropCols, binaryCols, catCols, 
    numericalCols, toPredict, 
    saveCSV, saveX, savey, 
    oheModelFile):

    print('Loading the data ...')
    if rawDataFormat == 'json':
        data = json.load(open(rawData))
        df = pd.DataFrame(data['result']['records'])
    elif rawDataFormat == 'csv':
        df = pd.read_csv(raw_data)
    else:
        print('Unknown data format:')
        return

    print('The columns are:')
    print(df.columns)

    # The numerical columns shoule be converted to floats
    for c in numericalCols:
        df[c] = df[c].astype(float)

    # Unnecessary columns should be dropped
    for c in dropCols:
        df.drop(c, axis=1, inplace=True)

    # The binary columns should be converted to 
    #  0 and 1
    for c in binaryCols:
        df[c] = (df[c] == binaryCols[c])*1

    # Categorical columns should be One Hot Encoded 
    if trainingMode:
        ohe = preprocessing.OneHotEncoder()
        ohe.fit( df[catCols].copy().values )
        dump( ohe, oheModelFile )
    else:
        print(f'Loading the model form [{oheModelFile}]')
        ohe = load( oheModelFile )

    oheVals = ohe.transform( df[catCols].copy().values ).toarray()
    df1     = pd.DataFrame( oheVals, columns=ohe.get_feature_names() )
    df      = pd.concat([df, df1], axis=1)

    for c in catCols:
        df.drop(c, axis=1, inplace=True)

    print('The first few lines of the data ..')
    print(df.head().T)
    
    y = df[toPredict].values
    X = df.drop(toPredict, axis=1).values

    print('Saving all the values ...')
    df.to_csv(saveCSV, index=False)
    np.save(saveX, X)
    np.save(savey, y)

    return

