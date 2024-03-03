import os
import shutil
import pandas as pd
import pdb

if __name__ == "__main__":
    df = pd.read_csv('./Chest_xray_Corona_Metadata.csv')

    for i in range(len(df)):
        path_to_get = './Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
        path_to_save = './imagefolder/'
        if df.Dataset_type[i] == 'TRAIN':
            path_to_get += 'train/'
            path_to_save += 'train/'
        else:
            path_to_get += 'test/'
            path_to_save += 'val/'
        if df.Label[i] == 'Normal':
            path_to_save += 'normal/'
        if df.Label_2_Virus_category[i] == 'COVID-19':
            path_to_save += 'viral-pneumonia'#'covid-19/' lump, since there's too few samples
        else:
            if df.Label_1_Virus_category[i] == 'bacteria':
                path_to_save += 'bacterial-pneumonia/'
            elif df.Label_1_Virus_category[i] == 'Virus':
                path_to_save += 'viral-pneumonia/'
            else:
                continue
        os.makedirs(path_to_save, exist_ok=True)
        path_to_get += df.X_ray_image_name[i]
        path_to_save += df.X_ray_image_name[i]
        shutil.copy(path_to_get, path_to_save)
    print("hi")
