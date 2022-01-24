import numpy as np
import pandas as pd
import itertools

import seaborn as sns
from matplotlib import pyplot as plt

current_folder = '/home/2014-0353_generaleye/Huy/YOLOv3/Main/Visualizations/'
input_file1 = '/home/2014-0353_generaleye/Huy/YOLOv3/Main/scores_all.xlsx'
input_file2 = 'PR_all.xlsx'

def create_metrics_plot(df_plot):
    with sns.plotting_context('paper', font_scale=1.3):
        g = sns.FacetGrid(df_plot, col='Metrics', hue='Model', col_wrap=2, palette='tab10', height=5, aspect=1.5, despine=False)
        g = g.map(sns.lineplot, 'Threshold', 'Score')

        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        g.add_legend(frameon=True)
        
    return g
    
def create_PR_curve(df_plot, df_F1_model):
    with sns.plotting_context('paper', font_scale=1.3):
        g = sns.FacetGrid(df_plot, col='Threshold', hue='Model', col_wrap=2, palette='tab10', height=5, aspect=1, despine=False)    
        for i in range(len(list_f1)):
            g = g.map(sns.lineplot, 'Recall', df_plot.columns[4+i], color='grey', alpha=0.1)
            txt_loc = (recall_cutoff-0.01, df_plot.loc[df_plot['Recall'] > recall_cutoff-0.01, df_plot.columns[4+i]].unique()+0.02)
            for ax in g.axes.ravel():
                ax.text(*txt_loc, df_plot.columns[4+i], horizontalalignment='right', size='small', color='grey', alpha=0.7)

        g = g.map(sns.lineplot, 'Recall', 'Precision')
        df_AP = df_plot.groupby(['Model', 'Threshold']).agg({'Precision': lambda x: np.nansum(x[0:101:10])/11}).reset_index()
        df_AP = df_AP.merge(df_F1_model, on=['Model', 'Threshold'])
        df_AP['Precision'] = df_AP['Precision'].round(2)
        df_AP['Model'] = pd.Categorical(df_AP['Model'], [x.lstrip('Model_') for x in list_models2])
        df_AP = df_AP.sort_values('Model')
        df_AP['Model'] = df_AP['Model'].replace({'COCO_Lab': 'COCO', 'Synth_Lab': 'Synth', 'Synth_DA_Lab': 'Synth_DA'})
        for ax, threshold in zip(g.axes.ravel(), df_plot['Threshold'].unique()):
            df_temp = df_AP.loc[df_AP['Threshold'] == threshold, ['Model', 'Precision', 'F1']]
            loc_legend = 'upper right' if threshold>=0.9 else 'lower left'
            ax.legend(loc=loc_legend,
                      handles=ax.lines[-4:], 
                      labels=['{}'.format(df_temp.iloc[i,0]) for i in range(len(df_temp))])
        g.set(ylim=(0, 1.05), xlim=(0, recall_cutoff))
        g.set_titles(row_template = '{row_name}', col_template = 'PR-Curve (IOU>0.5)')
        
    return g
    
    
if __name__ == '__main__':
    
    # INITIALIZE DATA SCORES
    data1 = pd.read_excel(input_file1, sheet_name=None)
    list_models = [model for model in list(data1) if 'Model' in model]
   
    # APPEND DATAFRAMES OF EACH MODEL
    df_plot = pd.DataFrame()
    for model in list_models:
        df_model = data1[model].iloc[:, :-1]
        df_pivot = df_model.melt(id_vars=['Threshold'], var_name='Metrics', value_name='Score')
        df_pivot['Model'] = model.lstrip('Model_')
        df_pivot = df_pivot[[df_pivot.columns[-1], *df_pivot.columns[:-1]]]

        df_plot = df_plot.append(df_pivot, ignore_index=True)

    # CREATE METRICS PLOT
    plot_metrics = create_metrics_plot(df_plot)
    plot_metrics.savefig(current_folder + 'Plot_metrics.png')

    # INITIALIZE DATA PR DATA
    data2 = pd.read_excel(current_folder + input_file2, sheet_name=None)

    # APPEND DATAFRAMES OF EACH MODEL
    list_output = {}
    df_F1_model = pd.DataFrame(columns={'Model', 'Threshold', 'F1'})
    list_models2 = [model for model in list(data2)]
    for model in list_models2:
        df_model = data2[model]
        
        df_model['F1'] = 2*df_model['Recall']*df_model['Precision']/(df_model['Precision'] + df_model['Recall']) 
        df_model['F1'] = df_model['F1'].fillna(0)
        df_model['Model'] = model.split('Model_')[-1]
        df_agg = df_model.groupby(['Model', 'Threshold']).agg({'F1': lambda x: np.max(x)}).reset_index()
        df_F1_model = df_F1_model.append(df_agg, ignore_index=True)

        df_output = pd.DataFrame()
        for threshold in [0.51, 0.6]:#, 0.7, 0.8]:
            df_temp = df_model[df_model['Threshold'] == threshold]

            p_interp = [df_temp.loc[df_model['Recall'] >= r, 'Precision'].max() for r in np.linspace(0,1,11)]
            df_RP_plot = pd.DataFrame({'Model': model.lstrip('Model_'), 'Threshold': threshold, 'Recall': np.linspace(0,1, 101),
                                       'Precision': [p_interp[0]] + list(itertools.chain(*[[p]*10 for p in p_interp[1:]]))})
            df_output = df_output.append(df_RP_plot, ignore_index=True)

        list_output[model] = df_output

    # CREATE PR CURVE
    df_plot = pd.concat(list_output, ignore_index=True)
    recall_cutoff = df_plot[(df_plot['Threshold'] == df_plot['Threshold'].min()) & (df_plot['Precision'].isnull())].groupby('Model').agg({'Recall': min}).max()[0] + 0.1
    df_plot = df_plot[df_plot['Recall'] <= recall_cutoff].fillna(0)
    list_f1 = [0.2, 0.4, 0.6, 0.8]
    for f1 in list_f1:
        f1_score = f1*df_plot['Recall'] / (2*df_plot['Recall']-f1)
        df_plot['f1={:.1f}'.format(f1)] = f1_score.apply(lambda x: x if x>=0 else np.nan)

    df_plot['F1'] = 2*df_plot['Recall']*df_plot['Precision']/(df_plot['Precision'] + df_plot['Recall'])    

    plot_PR_curve = create_PR_curve(df_plot, df_F1_model)
    plot_PR_curve.savefig(current_folder +'Plot_PR_curve.png')
    
