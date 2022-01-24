import pandas as pd
import numpy as np
import itertools

import os
import glob

from ast import literal_eval

current_folder = '/home/2014-0353_generaleye/Huy/YOLOv3/Main/Results_IMGLabels/'
    
def literal_converter(val):
    try:
        return literal_eval(str(val))   
    except:
        return []
    
def calculate_PR(df_input):
    df_eval = pd.DataFrame()
    for threshold in [i/100 for i in range(1, 101)]:
        df_filter = df_input.loc[df_input['Confidence'] >= threshold, :]

        df_temp = pd.DataFrame({'Confidence': [threshold], 'TP': [df_filter['TP'].sum()], 'FP': [df_filter['FP'].sum()]})
        df_temp['FN'] = len(df_input) - df_temp['TP']
        df_temp['Precision'] = np.round(df_temp['TP'] / (df_temp['TP'] + df_temp['FP']), 2)
        df_temp['Recall'] = np.round(df_temp['TP'] / (df_temp['TP'] + df_temp['FN']), 2)
        df_temp['F1'] = np.round(2*df_temp['Precision']*df_temp['Recall'] / (df_temp['Precision'] + df_temp['Recall']), 2)

        df_eval = df_eval.append(df_temp, ignore_index=True)

    df_output = pd.DataFrame({'Recall': np.linspace(0,1,101)})
    df_output['r_interp'] = np.floor(df_output['Recall']*10)/10
    df_output['Precision'] = [df_eval.loc[df_eval['Recall'] >= r, 'Precision'].max() for r in df_output['r_interp']]
    df_output['F1'] = np.round(2*df_output['Precision']*(df_output['r_interp']+0.1) / (df_output['Precision'] + (df_output['r_interp'])+0.1), 2)
    df_output = df_output.fillna(0)
    df_output = df_eval[['Confidence', 'Recall']].merge(df_output, on=['Recall'], how='right')
    
    AP = np.round(df_output.iloc[:, 1:].drop_duplicates()['Precision'].mean(), 3)

    return df_output, AP

def AP_table(df_input):
    dict_features = {'RGB': ['IR', 'RGB'], 'Low': ['High', 'Low'], 'Bed': ['No Bed', 'Bed'], 'Male': ['Female', 'Male']}
    df_output = pd.DataFrame(columns=['RGB', 'IR', 'Low', 'High', 'Bed', 'No Bed', 'Male', 'Female'],
                             index=['RGB', 'IR', 'Low', 'High', 'Bed', 'No Bed', 'Male', 'Female'])
    for feature_comb in itertools.permutations(list(dict_features), 2):
        for i,j in [(0,0), (0,1), (1,0), (1,1)]:
            df_filter = df_input.loc[(df_input[feature_comb[0]] == i) & (df_input[feature_comb[1]] == j), :]
            df_PR, AP = calculate_PR(df_filter)

            df_output.loc[dict_features[feature_comb[0]][i], dict_features[feature_comb[1]][j]] = AP.round(2)

    return df_output

def create_case_table(dict_input):
    dict_output = {}
    dict_case = {'Sensor': ['RGB', 'IR'], 'Occl.': ['Low', 'High'], 'Bed': ['Bed', 'No Bed'], 'Gender': ['Male', 'Female']}
    for case in list(dict_case):
        df_output = pd.DataFrame(columns=['RGB', 'IR', 'Low', 'High', 'Bed', 'No Bed', 'Male', 'Female'], index=list(dict_input))
        for model in list(dict_input):
            df_filter = dict_input[model].loc[dict_case[case], :]
            df_output.loc[model, :] = ['{:.2f} | {:.2f}'.format(i,j) for i,j in zip(df_filter.iloc[0, :], df_filter.iloc[1, :])]
            
        dict_output[case] = df_output.rename_axis('Model').reset_index()
            
    return dict_output

def AP_table_summary(df_input):
    list_features = ['RGB', 'Infrared', 'Low', 'High', 'Bed', 'No Bed', 'Male', 'Female']
    df_output = pd.DataFrame(columns=list_features)
    for feature in list_features:
        df_filter = df_input.loc[df_input[feature] == 1, :]
        df_PR, AP = calculate_PR(df_filter)
        
        df_output.loc[0, feature] = AP

    return df_output

def evaluate_imglabel(list_evaluations):
    df_output = pd.DataFrame(columns=['RGB', 'IR', 'Low', 'High', 'Bed', 'No Bed', 'Male', 'Female'])
    dict_eval = {}
    for file in list_evaluations:
        model_name = file.split('/')[-3]
        df_file = pd.read_csv(file, converters=dict.fromkeys(['bbox_true', 'bbox_test', 'Confidence', 'AoI', 'AoU', 'IoU'], literal_converter))
        
        df_file = df_file.merge(df_imglabel, on='IMG', how='right')
        
        df_flat = df_file.explode('bbox_true').explode('bbox_test')
        for col in ['Confidence', 'AoI', 'AoU', 'IoU']:
            df_flat.loc[-df_flat['bbox_test'].isnull(), col] = list(itertools.chain(*df_file[col].tolist()))
            df_flat[col] = df_flat[col].apply(lambda x: np.nan if isinstance(x, (list, tuple)) else x)
        
        df_flat['Low'] = (df_flat['Occlusion'] <= 20).astype(int)
        df_flat['High'] = (df_flat['Occlusion'] > 20).astype(int)
        df_flat['RGB'] = (df_flat['Infrared'] != 1).astype(int)
        df_flat['No Bed'] = (df_flat['Bed'] != 1).astype(int)
        
        dict_eval[model_name] = AP_table(df_flat)
        df_output.loc[model_name, :] = AP_table_summary(df_flat).values
        
    dict_output = create_case_table(dict_eval)
    df_output = df_output.rename_axis('Model').reset_index()
    
    return df_output, dict_output
    
if __name__ == '__main__':   
    df_imglabel = pd.read_csv('ArgusImages_test_IMGlabels.csv')
    list_evaluations = glob.glob('Model_*/Evaluation/*_evaluation.csv')

    df_output, dict_output = evaluate_imglabel(list_evaluations)
    df_output.to_csv(current_folder + 'Evaluation_IMGlabels.csv', index=False)
    
    with pd.ExcelWriter(current_folder + 'Evaluation_IMGlabels_case.xlsx') as writer:  
            for case in list(dict_output):
                df_eval_case = dict_output[case]
                df_eval_case.to_excel(writer, sheet_name=case, index=False)
                