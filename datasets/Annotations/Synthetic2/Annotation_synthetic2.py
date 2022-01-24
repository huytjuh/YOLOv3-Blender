import pandas as pd
import random
import glob

from cfg import *

def combine_annotation(list_txt, name_input, list_neg_txt=[]):
    with open('{}_Annotations.txt'.format(name_input), 'w') as out_file:
        for file in list_txt:
            with open(file) as f:
                anot = f.readlines()
            if anot[0][2:].split('/')[-1].rsplit(' ')[0].split('_')[-3] not in ['L00', 'L10']:
                out_file.write(base_path + anot[0][2:] + '\n')
        print('Number of synthetic samples: {}'.format(len(list_txt)))
    
        if list_neg_txt:
            for file in list_neg_txt[:20000]:
                with open(file) as f:
                    anot = f.readlines()
                out_file.write(anot[0].split(' ')[0] + '\n')
        print('Number of negative samples: {}'.format(len(list_neg_txt)))
    return 
    
    
if __name__ == '__main__':
    
    # LOAD CHARACTER IDS
    char_ids = pd.read_csv(base_path + 'Character_IDs.csv')
    
    # SPLIT & CREATE VALIDATION SET
    list_char_ids = {}
    if bool_val:
        sample = random.sample(range(len(char_ids)), k=int(len(char_ids)*frac_val))
        list_char_ids['Partial_Train'] = char_ids['Character'][~char_ids.index.isin(sample)].values.ravel()
        list_char_ids['Partial_Valid'] = char_ids['Character'][char_ids.index.isin(sample)].values.ravel()
    else:
        list_char_ids['Full_Train'] = char_ids['Character'].values.ravel()
        
    # LOOP OVER THE POSSIBLE ANNOTATION FILES
    for n in list(list_char_ids):
        list_id = list_char_ids[n]
    
        # OBTAIN INDIVIDUAL TXT ANNOTATIONS
        list_txt = []
        for char_id in list_id:
            list_txt.extend(glob.glob(img_path2 + 'RGB/**/{}_*.txt'.format(char_id)))
        
        # OBTAIN NEGATIVE SAMPLE TXT ANNOTATIONS
        list_neg_txt = []
        if n != 'Partial_Valid':
            list_neg_txt = glob.glob(img_path2 + neg_path + '*.txt')
        
        # WRITE COMBINED ANNOTATIONS
        combine_annotation(list_txt, n, list_neg_txt)