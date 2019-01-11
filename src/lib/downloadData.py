import urllib3
import os

def download(url, fileName, folder='data/raw_data'):
    '''download data and put that data within the specified file
    
    [description]
    
    Arguments:
        url {str} -- url from which to download JSON data
    
    Keyword Arguments:
        fileName {str} -- fileName to save the data in
        folder {str} -- folder to save the data in (default: {'../data/raw_data'})
    '''

    try:
        print('Download data from url: {}'.format(url))
        http = urllib3.PoolManager()
        data = http.request('GET', url)
        results = data.data.decode('utf-8')

        with open( os.path.join(folder, 'raw_data.json'), 'w' ) as fOut:
            fOut.write(results)

    except Exception as e:
        print(f'Unable to download data: {e}')

    return
