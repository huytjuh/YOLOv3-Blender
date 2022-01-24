import numpy as np
import pandas as pd
import glob
import itertools

from ast import literal_eval

import warnings
warnings.filterwarnings('ignore')

beta = 2

def literal_converter(val):
    try:
        return literal_eval(str(val))   
    except:
        return []

def calculate_PR(list_path):
    list_output = {}
    for file in list_path:
        model_name = file.split('/')[0]
        df_file = pd.read_csv(file, converters=dict.fromkeys(['bbox_true', 'bbox_test', 'Confidence', 'AoI', 'AoU', 'IoU'], literal_converter))
        print(model_name)
        df_flat = df_file.explode('bbox_true').explode('bbox_test')
        for col in ['Confidence', 'AoI', 'AoU', 'IoU']:
            df_flat.loc[-df_flat['bbox_test'].isnull(), col] = list(itertools.chain(*df_file[col].tolist()))
            df_flat[col] = df_flat[col].apply(lambda x: np.nan if isinstance(x, (list, tuple)) else x)
        df_flat = df_flat.sort_values('Confidence')
        
        df_output = pd.DataFrame()
        for overlap_threshold in np.linspace(0, 1, 101).round(2):    
            # DETERMINE TRUE POSITIVE, FALSE POSITIVE, FALSE NEGATIVE
            df_flat['TP'] = df_flat['IoU'].apply(lambda x: int(x >= overlap_threshold))
            df_flat['FP'] = df_flat['IoU'].apply(lambda x: int(x < overlap_threshold))
            df_flat.loc[(df_flat['bbox_true'].isnull()) & (df_flat['bbox_test'].notnull()), 'FP'] = 1

            # CALCULATE ACCUMULATED TP, FP, FN + PRECISION, RECALL (= FOR AP CALCULATIONS)
            df_flat = df_flat.sort_values('Confidence', ascending=False).reset_index(drop=True)
            df_flat['Acc_TP'] = df_flat['TP'].cumsum()
            df_flat['Acc_FP'] = df_flat['FP'].cumsum()
            df_flat['Acc_FN'] = len(df_flat) - df_flat['Acc_TP']
            df_flat['Precision'] = round(df_flat['Acc_TP'] / (df_flat['Acc_TP'] + df_flat['Acc_FP']), 4)
            df_flat['Recall'] = round(df_flat['Acc_TP'] / (df_flat['Acc_TP'] + df_flat['Acc_FN']), 4)    

            df_flat['Threshold'] = overlap_threshold
            df_output = df_output.append(df_flat[[df_flat.columns[-1]] + df_flat.columns[:-1].tolist()], ignore_index=True)

        list_output[model_name] = df_output

    return list_output

def calculate_metrics(list_PR):
    list_output = {}
    for model in list_PR:
        df_PR = list_PR[model]
        
        df_output = pd.DataFrame()
        for overlap_threshold in df_PR['Threshold'].unique():
            df_temp = df_PR[df_PR['Threshold'] == overlap_threshold]
            
            df_metrics = pd.DataFrame()
            df_metrics['Threshold'] = [overlap_threshold]
            df_metrics['Precision'] = [df_temp['TP'].sum() / (df_temp['TP'].sum() + df_temp['FP'].sum())]
            df_metrics['Recall'] = [df_temp['TP'].sum() / (df_temp['TP'].sum() + df_temp['Acc_FN'].values[-1])]
            df_metrics['Accuracy'] = [df_temp['TP'].sum() / (df_temp['TP'].sum() + df_temp['FP'].sum() + df_temp['Acc_FN'].values[-1])]
            df_metrics['F1_score'] = [2*df_metrics['Precision'][0]*df_metrics['Recall'][0] / (df_metrics['Precision'][0] + df_metrics['Recall'][0])]
            df_metrics['F{}_score'.format(beta)] = [(1 + beta**2)*df_metrics['Precision'][0]*df_metrics['Recall'][0] / ((beta**2)*df_metrics['Precision'][0] + df_metrics['Recall'][0])]
            
            df_output = df_output.append(df_metrics, ignore_index=True)
        list_output[model] = df_output
        
    return list_output
    
def get_summary(list_path):
    df_output = pd.DataFrame()
    for file in list_path:
        model_name = file.split('/')[0]
        
        temp = pd.read_csv(file)
        temp.index = [model_name] 
        
        df_output = df_output.append(temp)
    return df_output
    
    
if __name__ == '__main__':
    
    # OBTAIN PR-CURVE VALUES FOR ALL THRESHOLDS
    list_evaluations = glob.glob('Model_*/Evaluation/*_evaluation.csv')
    print([x.split('/')[0] for x in list_evaluations])
    list_PR = calculate_PR(list_evaluations)
    with pd.ExcelWriter('Visualizations/PR_all.xlsx') as writer:  
        for model in list(list_PR):
            df_PR = list_PR[model]
            df_PR.to_excel(writer, sheet_name=model, index=False)

    # OBTAIN METRICS FOR ALL THRESHOLDS
    list_metrics = calculate_metrics(list_PR)
    with pd.ExcelWriter('scores_all.xlsx') as writer:  
        for model in list(list_metrics):
            df_metrics = list_metrics[model]
            df_metrics.to_excel(writer, sheet_name=model, index=False)

    # OBTAIN SUMMARY
    list_scores = glob.glob('Model_*/Evaluation/*_scores.csv')
    df_summary = get_summary(list_scores)
    with pd.ExcelWriter('scores_all.xlsx', mode='a') as writer: 
        df_summary.to_excel(writer, sheet_name='Summary')
