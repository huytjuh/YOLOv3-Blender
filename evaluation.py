import numpy as np         
import pandas as pd
import glob
import itertools
from ast import literal_eval                                 

from cfg import *

test_folder = 'Output/'
true_folder = true_path
overlap_threshold = 0.5
beta = 2

def intersection_area(A, B):
    if pd.isnull(A) | pd.isnull(B):
        return np.nan
    bbox_A = list(map(int, A.split(',')[:-1]))
    bbox_B = list(map(int, B.split(',')[:-1]))
    x1, y1 = max(bbox_A[0], bbox_B[0]), max(bbox_A[1], bbox_B[1])
    x2, y2 = min(bbox_A[2], bbox_B[2]), min(bbox_A[3], bbox_B[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    AoI = (x2 - x1) * (y2 - y1)
    return AoI
    
def union_area(A,B):
    if pd.isnull(A) | pd.isnull(B):
        return np.nan
    
    bbox_A = list(map(int, A.split(',')[:-1]))
    bbox_B = list(map(int, B.split(',')[:-1]))
    area_A = (bbox_A[2] - bbox_A[0]) * (bbox_A[3] - bbox_A[1])
    area_B = (bbox_B[2] - bbox_B[0]) * (bbox_B[3] - bbox_B[1])
    AoI = intersection_area(A, B)
    AoU = area_A + area_B - AoI
    return AoU

def calculate_PR(df_input):
    df_eval = pd.DataFrame()
    for threshold in [i/100 for i in range(1, 101)]:
        df_filter = df_input.loc[df_input['Confidence'] >= threshold, :]

        df_temp = pd.DataFrame({'Confidence': [threshold], 'TP': [df_filter['TP'].sum()], 'FP': [df_filter['FP'].sum()]})
        df_temp['FN'] = len(df_input) - df_temp['TP']
        df_temp['Precision'] = np.round(df_temp['TP'] / (df_temp['TP'] + df_temp['FP']), 3)
        df_temp['Recall'] = np.round(df_temp['TP'] / (df_temp['TP'] + df_temp['FN']), 3)
        df_temp['F1'] = np.round(2*df_temp['Precision']*df_temp['Recall'] / (df_temp['Precision'] + df_temp['Recall']), 3)

        df_eval = df_eval.append(df_temp, ignore_index=True)

    df_output = pd.DataFrame({'Recall': np.linspace(0,1,101)})
    df_output['r_interp'] = np.floor(df_output['Recall']*10)/10
    df_output['Precision'] = [df_eval.loc[df_eval['Recall'] >= r, 'Precision'].max() for r in df_output['r_interp']]
    df_output['F1'] = np.round(2*df_output['Precision']*(df_output['r_interp']+0.1) / (df_output['Precision'] + (df_output['r_interp'])+0.1), 3)
    df_output = df_output.fillna(0)
    df_output = df_eval[['Confidence', 'Recall']].merge(df_output, on=['Recall'], how='right')
    
    AP = np.round(df_output.iloc[:, 1:].drop_duplicates()['Precision'].mean(), 3)

    return df_output, AP

if __name__ == '__main__':
    # OBTAIN LIST OF BOUNDING BOXES
    list_test = glob.glob(test_folder + '*.txt')
    list_true = glob.glob(true_folder + '*.txt')
    
    # SELECT ONLY THE RELEVANT TRUE BOUNDING BOX >> LEN(LIST_TEST) == LEN(LIST_TRUE)
    true_test = [x.split('/')[-1] for x in list_test]
    list_true = [x for x in list_true if x.split('/')[-1] in true_test]
    
    # EXTRACT TEST BOUNDING BOX
    df_test = pd.DataFrame()
    for path in list_test: 
        with open(path) as file:
            line = file.readline()
        if line.split(' ')[-1]:
            bbox = [box.rstrip('\n') for box in line.split(' ')[1:] if box.split(',')[-1] == 'person']
        else:
            bbox = [['0,0,0,0,0']]
        temp = {'IMG': line.split(' ')[0].split('/')[-1].strip(), 'bbox_test': bbox}
        df_test = df_test.append(temp, ignore_index=True)
    
    # EXTRACT TRUE BOUNDING BOX
    df_true = pd.DataFrame()
    for path in list_true:
        with open(path) as file:
            line = file.readline()
        if line.split(' ')[1:][-1].rstrip('\n'):
            bbox = [box.rstrip('\n') for box in line.split(' ')[1:] if box.rstrip('\n')[-1] == '0']
        else:
            bbox = ['0,0,0,0,0']
        temp = {'IMG': line.split(' ')[0].split('/')[-1].strip(), 'bbox_true': bbox}
        df_true = df_true.append(temp, ignore_index=True)
    
    df_output = df_true.merge(df_test, on='IMG', how='left')

    # EXTRACT CONFIDENCE SCORE
    data_conf = pd.read_csv(test_folder + 'confidence_scores.csv', converters={'bbox_test': literal_eval, 'Confidence': literal_eval})
    df_conf = data_conf.explode('bbox_test')
    df_conf.loc[-df_conf['bbox_test'].isnull(), 'Confidence'] = list(itertools.chain(*data_conf['Confidence'].tolist()))
    df_conf['Confidence'] = df_conf['Confidence'].apply(lambda x: np.nan if isinstance(x, (list, tuple)) else x)

    df_output = df_output.explode('bbox_true').explode('bbox_test')
    df_output = df_output.merge(df_conf, on=['IMG', 'bbox_test'], how='left')

    # EVALUATE TEST BOUNDING BOX VS TRUE BOUNDING BOX
    df_output['AoI'] = df_output.apply(lambda x: intersection_area(x['bbox_true'], x['bbox_test']), axis=1)
    df_output['AoU'] = df_output.apply(lambda x: union_area(x['bbox_true'], x['bbox_test']), axis=1)
    df_output['IoU'] = round(df_output['AoI'] / df_output['AoU'], 4)
    
    # DETERMINE TRUE POSITIVE, FALSE POSITIVE, FALSE NEGATIVE
    df_output['TP'] = df_output['IoU'].apply(lambda x: int(x >= overlap_threshold))
    df_output['FP'] = df_output['IoU'].apply(lambda x: int(x < overlap_threshold))
    df_output.loc[(df_output['bbox_true'].isnull()) & (df_output['bbox_test'].notnull()), 'FP'] = 1
    
    df_output = df_output.sort_values('Confidence', ascending=False).reset_index(drop=True)
    df_output['Acc_TP'] = df_output['TP'].cumsum()
    df_output['Acc_FP'] = df_output['FP'].cumsum()
    df_output['Acc_FN'] = len(df_output) - df_output['Acc_TP']
    df_output['Precision'] = round(df_output['Acc_TP'] / (df_output['Acc_TP'] + df_output['Acc_FP']), 3)
    df_output['Recall'] = round(df_output['Acc_TP'] / (df_output['Acc_TP'] + df_output['Acc_FN']), 3) 
    df_output['r_interp'] = np.floor(df_output['Recall']*10)/10
    df_output['F1'] = np.round(2*df_output['Precision']*df_output['Recall'] / (df_output['Precision'] + df_output['Recall']), 3)
    
    # SAVE COMPRESSED FORMAT
    df_compressed = df_output.groupby('IMG').apply(lambda x: pd.DataFrame([[x[col].tolist()] for col in df_output.columns[1:7].tolist()], 
                                                                          index=df_output.columns[1:7].tolist()).T).reset_index(0)
    df_compressed['bbox_true'] = df_compressed['bbox_true'].apply(lambda x: np.unique(x))                                                                                    
    df_TP_FP = df_output.groupby('IMG').apply(lambda x: pd.DataFrame([[x['TP'].sum(), x['FP'].sum()]], columns=['TP', 'FP'])).reset_index(0)
    df_compressed = df_compressed.merge(df_TP_FP, on='IMG', how='left')
    df_compressed.to_csv('Evaluation/{}_evaluation.csv'.format(model_name), index=False)
    
    df_metrics, AP = calculate_PR(df_output)
    
    #df_metrics = df_metrics.loc[df_metrics['F1'].idxmax(), ['Confidence', 'Recall', 'Precision', 'F1']]
    df_metrics = df_output.loc[df_output['F1'].idxmax(), ['Confidence', 'Recall', 'r_interp', 'Precision', 'F1']]
    df_metrics['AP interp'] = AP
    df_metrics['AP'] = np.nansum([df_output.loc[df_output['Recall'] >= r, 'Precision'].max() for r in np.linspace(0,1,11)]) / 11
    
    print(df_metrics)