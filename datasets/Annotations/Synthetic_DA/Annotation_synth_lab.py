import pandas as pd
import random
import glob

from cfg import *

def combine_annotation(list_txt, name_input, annot_path, list_neg_txt=[]):
    with open('{}_Annotations.txt'.format(name_input), 'w') as out_file:
        for file in list_txt[:50000]:
            with open(file) as f:
                anot = f.readline()
            if len(anot.rsplit('\n')[0].split(' ')[1:]) > 0:
                out_file.write(file.split('.')[0] + '.' + anot.split(' ')[0].split('.')[-1] + ' ' + ' '.join(anot.split(' ')[1:]).rsplit('\n')[0] + '\n')
        print('Number of {} samples: {}'.format(name_input, len(list_txt[:50000])))
    
        if list_neg_txt:
            for file in list_neg_txt[:len(list_txt)]:
                with open(file) as f:
                    anot = f.readline()
                out_file.write(file.split('.')[0] + '.' + anot.split(' ')[0].split('.')[-1] + ' ' + ' '.join(anot.split(' ')[1:]).rsplit('\n')[0] + '\n')
        print('Number of negative samples: {}'.format(len(list_neg_txt[:len(list_txt)])))
    return 
    
    
if __name__ == '__main__':
    list_dataset = {'Synth': {'glob': img_path + '**/*.txt', 'annot': img_path}}
    if bool_val:
        list_dataset['Lab'] = {'glob': val_path + '*.txt', 'annot': val_path}

    for dataset in list_dataset:
        glob_path = list_dataset[dataset]['glob']
        annot_path = list_dataset[dataset]['annot']
    
        # OBTAIN INDIVIDUAL TXT ANNOTATIONS
        list_txt = glob.glob(glob_path)
        
        # OBTAIN NEGATIVE SAMPLE TXT ANNOTATIONS
        list_neg_txt = []
        if bool_neg:
            list_neg_txt = glob.glob(base_path + 'Output_IMG/' + neg_path + '*.txt')
        
        # WRITE COMBINED ANNOTATIONS
        combine_annotation(list_txt, dataset, annot_path, list_neg_txt)
    
        