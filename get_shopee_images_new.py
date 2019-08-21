from os.path import join, isfile, isdir
import os
import urllib.request
import pandas as pd


def main():
    data_folder = "data/qc/20190805/images/"
    if not isdir(data_folder):
        os.mkdir(data_folder)
    in_file = "data/qc/crop_20190805.csv"
    prefix = "http://cf.shopee.co.id/file/"
    df = pd.read_csv(in_file)
    data = df.head(100)

    for index, record in data.iterrows():
        url = prefix + record['hash']
        print(url)
        filename = os.path.join(data_folder, record['hash'] + '.jpg')
        urllib.request.urlretrieve(url, filename)

if __name__ == '__main__':
    main()