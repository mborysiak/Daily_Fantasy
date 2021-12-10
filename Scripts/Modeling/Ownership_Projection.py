import pandas as pd
import os
import zipfile
import numpy as np
download_path = '//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/Mborysiak/DK'
extract_path = download_path + 'Results/'

csv_files = [f for f in os.listdir(extract_path)]
df = pd.DataFrame()
for f in csv_files:
    df = pd.concat([df, pd.read_csv(extract_path+f, low_memory=False)[['Points', 'Rank']]], axis=0)
