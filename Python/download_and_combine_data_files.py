# This script will combine the decompressed data files into a single file for easier use going forward. It will also determine the need for multiple data dictionaries and potentially create a single dictionary for all of the data.

import numpy as np
import pandas as pd
import glob

project_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/' # Local location of project directory
data_fp = project_fp + 'Data/Raw_data/' # subdirectory containing NTDB data, each year (2007-2022) listed as: PUF AY *YEAR*

# Files were downloaded as zip files and decompressed. This created directories containing: .csv, .sas7bdat, .xlsx and .pdf files. These files consist of raw data (.csv and .sas7bdat), data dictionary for associated year (.pdf) and corresponding data dictionary (.xlsx) [notable, the .xlsx file listed is for a range of years that does not cover all years being studied; Will need to verify that these are the same otherwise will have to process each year set with a different dictionary and then create a new common dictionary that can be used for all data when combined together]. Finally there is also a user manual for the associated year range (.pdf)

years = range(2007, 2023)

combined_years_df = pd.DataFrame()

for i in years:
    if i in range(2007, 2009):
        print('Year: %i'%i)
        year_fp = data_fp +'PUF AY %i/'%i
        csv_fp = year_fp + 'CSV/'
        csv_files = glob.glob(csv_fp + '*.csv')
        year_df = pd.DataFrame()
        for x in csv_files:
            df = pd.read_csv(x).head()
            if 'INC_KEY' not in list(df.columns.tolist()):
                print('%s has NO INC_KEY value'%x)
                print()
            #year_df = pd.merge(year_df, df, 
            #on='INC_KEY',
            #how='outer')
            
        
        
    #elif i in range(2010, 2016):
        
        
    #elif i == 2016:
        
        
    #elif i in range(2017,2019):
        
        
    #elif i in range(2019,2023):
        


