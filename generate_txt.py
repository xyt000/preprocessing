from os import walk, path
import pandas as pd

if __name__ == '__main__':

    landmark_names = ['mandible dentry', 'hyoid fusion', 'first vertebra', 'optic nerve head R',
                      'optic nerve head L']

    task = 'images'
    with open(f'medaka_{task}_2_0.txt', 'w') as tg_file:
        for dirpath, dirnames, filenames in walk('/home/ws/ml0077/work/data/medaka/augmentation/data2'):
            for f in sorted(filenames):
                if f.endswith('.csv') and task == 'landmarks':
                    dat = pd.read_csv(path.join(dirpath, f))
                    with open(path.join(dirpath, f.replace('.csv', '.txt')), 'w') as txt_file:
                        for lm in landmark_names:
                            txt_file.write(f'{dat.loc[0, lm]},{dat.loc[1, lm]},{dat.loc[2, lm]},\n')
                    tg_file.write(path.join(dirpath, f.replace('.csv', '.txt')))
                    tg_file.write('\n')
                elif f.endswith('.tif') and task == 'images':
                    tg_file.write(path.join(dirpath, f))
                    tg_file.write('\n')