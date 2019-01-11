import json

from lib import downloadData as dD 
from lib import featureEngg as fE

from lib.regressors import linearRegression as reg_LR
from lib.regressors import ridgeRegression  as reg_ridge

def main():

    try:
        config = json.load(open('config/config.json'))

        if 'toDos' in config:

            if config['toDos']['downloadData']:
                try:
                    print('Downloading data ...')
                    dD.download(**config['downloadData'])
                except Exception as e:
                    print('Unable to download data: {}'.format(e))

            if config['toDos']['featureEngg']:
                try:
                    print('Doing Feature Engineering ...')
                    fE.doFeatureEngineering(**config['featureEngg'])
                except Exception as e:
                    print('Unable to download data: {}'.format(e))

            if config['toDos']['linearRegression']:
                try:
                    print('Generating a Linear Model ...')
                    reg_LR.doLinearRegression( **config['linearRegression'] )
                except Exception as e:
                    print(f'Unable to do Linear Regression: {e}')
            
            if config['toDos']['ridgeRegression']:
                try:
                    print('Generating a Ridge Model ...')
                    reg_ridge.doLinearRegression( **config['ridgeRegression'] )
                except Exception as e:
                    print(f'Unable to do Ridge Regression: {e}')

    except Exception as e:
        print('Problem with program execution: {}'.format(e))

    return

if __name__ == '__main__':
    main()

