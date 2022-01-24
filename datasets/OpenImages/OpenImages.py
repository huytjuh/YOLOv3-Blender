import pandas as pd
import numpy as np
import time as dt
import random
import json
import argparse
import os
import subprocess

from flatten_json import flatten

import cv2
from PIL import Image

from cfg import *

def json_to_df(list_categories, column='Category'):
    dict_flattened = (flatten(record, '_') for record in list_categories['Subcategory'])
    df_categories = pd.DataFrame(dict_flattened)
    for i in df_categories.columns:
        df_categories.loc[:, i] = df_categories.loc[:, i].map(labels.set_index('LabelName').T.to_dict('records')[0])

    df_categories = df_categories[df_categories.columns[~df_categories.columns.str.contains(*[x for x in ['Part', 'Subcategory'] if x != column])]]
    df_categories = df_categories[['LabelName', *df_categories.columns[1:].sort_values()]]
    df_categories = df_categories[-df_categories.iloc[:, 1:].isnull().all(axis=1)].reset_index(drop=True)

    df_output = pd.DataFrame()
    for i in range(len(df_categories)):
        temp = df_categories.iloc[i, :][-df_categories.iloc[i, :].isnull()]
        max_nr = max(list(map(int, [x[1] for x in temp.index[1:].str.split('_').tolist()])))
        df_temp = pd.DataFrame()
        for j in range(max_nr+1):
            df_input = temp[temp.index.str.startswith('{}_{}_'.format(column, j))]
            if len(df_input) >  1:
                df_input = pd.DataFrame({'LabelName': df_input[1:], column: df_input[0]}).reset_index(drop=True)
            else:
                df_input = pd.DataFrame({'LabelName': [df_input[0]], column: [temp[0]]})
            df_temp = df_temp.append(df_input, ignore_index=True)
        df_output = df_output.append(df_temp, ignore_index=True)

    return df_output.drop_duplicates()


if __name__ == '__main__':

    # LOAD SET OF IMAGEIDS
    data = pd.read_csv(base_path + 'oidv6-train-annotations-bbox.csv')

    # LOAD SET OF LABELS
    labels = pd.read_csv(base_path + 'class-descriptions-boxable.csv', header=None)
    labels.columns = ['LabelName', 'Object']

    # MAP LABELS TO CORRECT IMAGEID 
    data['LabelName'] = data['LabelName'].map(labels.set_index('LabelName').T.to_dict('records')[0])

    # LOAD JSON FILE CONTAINING CATEGORIES/HIERARCHY
    with open(base_path + 'bbox_labels_600_hierarchy.json') as json_file:
        list_categories = json.load(json_file)
    
    # TRANSFORM JSON TO DATAFRAME
    df_cate = json_to_df(list_categories, 'Subcategory').rename(columns={'Subcategory': 'Category'})  # HYPERNYMS
    df_part = json_to_df(list_categories, 'Part').rename(columns={'Part': 'Category'})                # MERONYMS
    df_group = pd.concat([df_cate, df_part]).reset_index(drop=True)

    ### FILTER OUT PERSON LABELS BASED ON JSON CATEGORIES ###
    
    # MAP CATEGORIES TO THEIR LABELNAME
    df = data.copy()
    df['Category'] = df['LabelName'].map(df_group.set_index('LabelName').T.to_dict('records')[0])
    print(df['Category'].unique())

    # FILTER OUT IMAGEID CONTAINING LABELS
    list_filter = ['Person', 'Human body']
    del_obs = df.loc[(df['LabelName'].isin(list_filter)) | (df['Category'].isin(list_filter)), 'ImageID'].unique()
    if bool_person == True:
        df = df[df['ImageID'].isin(del_obs)]
    else:
        df = df[-df['ImageID'].isin(del_obs)]
    print('ImageIDs filtered out: {} out of {}'.format(len(del_obs), len(data['ImageID'].unique())))

    # OBTAIN RANDOMLY SAMPLED IMAGEID WITHOUT PERSON
    random.seed(1234)
    sample = random.sample(df['ImageID'].unique().tolist(), k=n_samples)
    df_sample = df[df['ImageID'].isin(sample)].reset_index(drop=True)
    print(df_sample['ImageID'])
    
    ### DOWNLOAD OPEN IMAGES FROM GOOGLE CLOUD API ###
    
    # CREATE TXT FILE CONSISTING OF IMAGEIDS
    df_images = pd.Series('train/' + df_sample['ImageID'].unique())
    df_images.to_csv(base_path + 'Model/txt_images.txt', sep='\t', index=False, header=False)

    if False:
        # EXECUTE SHELL COMMAND TO DOWNLOAD OPENIMAGES #
        subprocess.call(['python3', 'Model/downloader.py', 'Model/txt_images.txt', 
                         '--download_folder', output_path[:-1],
                         '--num_processes', str(n_cores)])

    ### OBTAIN .TXT FILES CONTAINING LABELS IN YOLOV3 FORMAT ###

    # CREATE .TXT FILES CONTAINING CLASSES
    df_names = pd.Series(df_sample['LabelName'].unique())
    df_names.to_csv(base_path + 'Model/OpenImages_classes.txt', header=False)

    # CREATE TXT NOTATION FILE FOR EACH IMAGE
    for imageID in df_sample['ImageID'].unique():
        image = Image.open(output_path + '{}.jpg'.format(imageID))
        size = image.size

        temp = df_sample[df_sample['ImageID'] == imageID]
        temp.loc[:, ['XMin', 'XMax']], temp.loc[:, ['YMin', 'YMax']] = temp.loc[:, ['XMin', 'XMax']].round(4)*size[0], temp.loc[:, ['YMin', 'YMax']].round(4)*size[1]
        temp.loc[:, ['XMin', 'XMax']], temp.loc[:, ['YMin', 'YMax']] = temp.loc[:, ['XMin', 'XMax']].astype('int'), temp.loc[:, ['YMin', 'YMax']].astype('int')
        df_input = pd.DataFrame({'x_min': temp['XMin'], 'y_min': temp['YMin'], 'x_max': temp['XMax'], 'y_max': temp['YMax'], 'class': df_names[df_names==temp['LabelName'].values[0]].index[0]}).reset_index(drop=True)
        
        with open(output_path + '{}.txt'.format(imageID), 'w') as out_file:
          out_file.write(output_path + '{}.jpg'.format(imageID))
          for i in range(len(df_input)):
            box = '{},{},{},{},{}'.format(df_input.loc[i, 'x_min'], df_input.loc[i, 'y_min'], df_input.loc[i, 'x_max'], df_input.loc[i, 'y_max'], 1)
            out_file.write(' ' + box)