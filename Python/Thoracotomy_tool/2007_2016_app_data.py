import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

data_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/Data/Raw_data/PUF AY %i/CSV/'
years = range(2007, 2017)
thoracotomy_code = 34.02
sternotomy_codes = ['0P800ZZ']

mechanism_code_dict = {1:'Cut/pierce', 2:'Drowning/submersion', 3:'Fall', 4:'Fire/flame', 5:'Hot object/substance', 6:'Firearm', 7:'Machinery', 8:'MVT Occupant', 9:'MVT Motorcyclist', 10:'MVT Pedal cyclist', 11:'MVT Pedestrian', 12:'MVT Unspecified', 13:'MVT Other', 14:'Pedal cyclist, other', 15:'Pedestrian, other', 16:'Transport, other', 17:'Natural/environmental,  Bites and stings', 18:'Natural/environmental,  Other', 19:'Overexertion', 20:'Poisoning', 21:'Struck by, against', 22:'Suffocation', 23:'Other specified and classifiable', 24:'Other specified, not elsewhere classifiable', 25:'Unspecified', 26:'Adverse effects, medical care', 27:'Adverse effects, drugs'} # As noted in PUF dictionary
trauma_type_code_dict = {1:'Blunt', 2:'Penetrating', 3:'Burn', 4:'Other/unspecified', 9:'Activity Code - Not Valid as a Primary E-Code'} # As noted in PUF Dictionary
sex_code_dict = {1:'Male', 2:'Female'}
eddischarge_code_dict = {1: 'Floor bed (general admission, non-specialty unit bed)', 2: 'Observation unit (unit that provides < 24 hour stays)', 3: 'Telemetry/step-down unit (less acuity than ICU)', 4: 'Home with services', 5: 'Deceased/expired', 6: 'Other (jail, institutional care, mental health, etc.)', 7: 'Operating Room', 8: 'Intensive Care Unit (ICU)', 9: 'Home without services', 10: 'Left against medical advice', 11: 'Transferred to another hospital'}
hospdischarge_disposition_code_dict = {1: 'Discharged/Transferred to a short-term general hospital for inpatient care', 2: 'Discharged/Transferred to an Intermediate Care Facility (ICF)', 3: 'Discharged/Transferred to home under care of organized home health service', 4: 'Left against medical advice or discontinued care', 5: 'Deceased/Expired', 6: 'Discharged to home or self-care (routine discharge)', 7: 'Discharged/Transferred to Skilled Nursing Facility (SNF)', 8: 'Discharged/Transferred to hospice care', 10: 'Discharged/Transferred to court/law enforcement', 11: 'Discharged/Transferred to inpatient rehab or designated unit', 12: 'Discharged/Transferred to Long Term Care Hospital (LTCH)', 13: 'Discharged/Transferred to a psychiatric hospital or psychiatric distinct part unit of a hospital', 14: 'Discharged/Transferred to another type of institution not defined elsewhere'}

input_list = ['SEX', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE']

for year in years:
    print(year)
    if year in range(2010, 2017, 1):
        PCODE_df = pd.read_csv(data_fp%year + 'PUF_PCODE.csv').sort_values('INC_KEY')
        PCODE_df = PCODE_df.rename(columns={'HOURTOPROC': 'HOURTOPRO'})
    else:
        PCODE_df = pd.read_csv(data_fp%year + 'PUF_PCODE.csv').sort_values('INC_KEY')
    PCODEDES_df = pd.read_csv(data_fp%year + 'PUF_PCODEDES.csv')
    DISCHARGE_df = pd.read_csv(data_fp%year + 'PUF_DISCHARGE.csv', usecols=['INC_KEY', 'HOSPDISP', 'LOSMIN', 'LOSDAYS', 'ICUDAYS', 'VENTDAYS']).sort_values('INC_KEY')
    ECODE_df = pd.read_csv(data_fp%year + 'PUF_ECODE.csv', usecols=['INC_KEY', 'ECODE']).sort_values('INC_KEY')
    ECODEDES_df = pd.read_csv(data_fp%year + 'PUF_ECODEDES.csv', usecols=['INJTYPE', 'MECHANISM', 'ECODE', 'ECODEDES'])
    DEMO_df = pd.read_csv(data_fp%year + 'PUF_DEMO.csv', usecols=['INC_KEY', 'GENDER']).sort_values('INC_KEY')
    VITALS_df = pd.read_csv(data_fp%year + 'PUF_VITALS.csv', usecols=['INC_KEY', 'VSTYPE', 'SBP', 'PULSE', 'RR', 'OXYSAT', 'GCSTOT']).sort_values('INC_KEY')
    VITALS_df = VITALS_df[VITALS_df['VSTYPE'] == 'EMS']
    if year in range(2013, 2017, 1):
        PM_df = pd.read_csv(data_fp%year + 'PUF_PM.csv', usecols=['inc_key', 'HEMORRHAGE_CTRL_STYPE_CODE', 'HEMORRHAGE_CTRL_STYPE_DESC', 'HemorrhageCtrlMins', 'HemorrhageCtrlDays']).sort_values('inc_key')
        PM_df.columns = map(str.upper, PM_df.columns)
    else:
        pass
    
    if year in range(2007, 2011, 1):
        ED_df = pd.read_csv(data_fp%year + 'PUF_ED.csv', usecols=['INC_KEY', 'EDDISP', 'EDDEATH']).sort_values('INC_KEY')
    elif year in range(2011, 2017, 1):
        ED_df = pd.read_csv(data_fp%year + 'PUF_ED.csv', usecols=['INC_KEY', 'EDDISP', 'SIGNSOFLIFE']).sort_values('INC_KEY')
        ED_df = ED_df.rename(columns={'SIGNSOFLIFE':'EDDEATH'})
    else:
        pass
    
    # Force column headers to be all upper case and fill in missing year data
    
    PCODE_df.columns = map(str.upper, PCODE_df.columns)
    ED_df.columns = map(str.upper, ED_df.columns)
    PCODEDES_df.columns = map(str.upper, PCODEDES_df.columns)
    PCODE_df['YOPROC'] = PCODE_df['YOPROC'].fillna(year)
    VITALS_df.columns = map(str.upper, VITALS_df.columns)
    
    PCODE_df['PROCEDUREMINS'] = PCODE_df['HOURTOPRO']*60
    
    PCODE_df = PCODE_df.loc[:, ['INC_KEY', 'PCODE', 'DAYTOPROC', 'PROCEDUREMINS', 'HOURTOPRO']]
    
    ECODE_LOOKUP_df = ECODEDES_df.loc[:, ['ECODE', 'ECODEDES', 'MECHANISM', 'INJTYPE']]
    
    print(np.shape(PCODE_df))
    
    thoracotomy_list = PCODE_df.loc[PCODE_df['PCODE'] == thoracotomy_code]
    
    print(np.shape(thoracotomy_list))

    if year in range(2013, 2017, 1):
        DCS_list = PM_df.loc[PM_df['HEMORRHAGE_CTRL_STYPE_CODE'] == 3.0]
        thoracotomy_keys = np.unique(DCS_list['INC_KEY'].tolist())
        print(np.shape(thoracotomy_keys))
    else:
        thoracotomy_keys = np.unique(thoracotomy_list['INC_KEY'].tolist())
        print(np.shape(thoracotomy_keys))

    
    PCODE_df = PCODE_df.loc[PCODE_df['INC_KEY'].isin(thoracotomy_keys)]
    ECODE_df = ECODE_df.loc[ECODE_df['INC_KEY'].isin(thoracotomy_keys)]
    DEMO_df = DEMO_df.loc[DEMO_df['INC_KEY'].isin(thoracotomy_keys)]
    ED_df = ED_df.loc[ED_df['INC_KEY'].isin(thoracotomy_keys)]
    VITALS_df = VITALS_df[VITALS_df['INC_KEY'].isin(thoracotomy_keys)]
    DISCHARGE_df = DISCHARGE_df.loc[DISCHARGE_df['INC_KEY'].isin(thoracotomy_keys)]
    
    print(np.shape(ECODE_df))
    
    procedure_dict = dict(zip(PCODEDES_df['PCODE'], PCODEDES_df['PCODEDESCR'])) # create dictionary
    PCODE_df['PCODE'] = PCODE_df['PCODE'].replace(procedure_dict) # implement dictionary
    
    ecode_dict = dict(zip(ECODEDES_df['ECODE'], ECODEDES_df['ECODEDES'])) # create dictionary
    ECODE_df['ECODE'] = ECODE_df['ECODE'].replace(ecode_dict) # implement dictionary
    
    print(np.shape(ECODE_df))
    
    mechanism_dict = dict(zip(ECODEDES_df['ECODEDES'], ECODEDES_df['MECHANISM'])) # create dictionary
    ECODE_df['MECHANISM'] = ECODE_df['ECODE'].map(mechanism_dict) # implement dictionary
    
    print(np.shape(ECODE_df))
    
    traumatype_dict = dict(zip(ECODEDES_df['ECODEDES'], ECODE_LOOKUP_df['INJTYPE'])) # create dictionary
    ECODE_df['INJTYPE'] = ECODE_df['ECODE'].map(traumatype_dict) # implement dictionary
    
    print(np.shape(ECODE_df))
    
    ECODE_df['MECHANISM'] = ECODE_df['MECHANISM'].replace(mechanism_code_dict) # implement dictionary, may not need
    
    print(np.shape(ECODE_df))
    
    ECODE_df['INJTYPE'] = ECODE_df['INJTYPE'].replace(trauma_type_code_dict) # implement dictionary, may not need
    
    print(np.shape(ECODE_df))
    
    DEMO_df['GENDER'] = DEMO_df['GENDER'].replace(sex_code_dict) # May not need
    
    print(np.shape(DEMO_df))
    
    ED_df['EDDISP'] = ED_df['EDDISP'].replace(eddischarge_code_dict) # implement dictionary
    
    print(np.shape(ED_df))
    
    DISCHARGE_df['HOSPDISP'] = DISCHARGE_df['HOSPDISP'].replace(hospdischarge_disposition_code_dict) # implement dictionary

    if year in range(2013, 2017, 1):
        DCS_df = PM_df.loc[PM_df['INC_KEY'].isin(thoracotomy_keys)]
        TRAUMA_df = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(ECODE_df, DEMO_df, on='INC_KEY', how='left'), ED_df, on='INC_KEY', how='left'), DISCHARGE_df, on='INC_KEY', how='left'), DCS_df, on='INC_KEY', how='left'), VITALS_df, on='INC_KEY', how='left')
    elif year in range(2007, 2013, 1):
        ECODE_df = ECODE_df.assign(**{'HEMORRHAGE_CTRL_STYPE_CODE': np.nan, 'HEMORRHAGE_CTRL_STYPE_DESC': np.nan, 'HEMORRHAGECTRLMINS': np.nan, 'HEMORRHAGECTRLDAYS': np.nan})
        TRAUMA_df = pd.merge(pd.merge(pd.merge(pd.merge(ECODE_df, DEMO_df, on='INC_KEY', how='left'), ED_df, on='INC_KEY', how='left'), DISCHARGE_df, on='INC_KEY', how='left'), VITALS_df, on='INC_KEY', how='left')
    TRAUMA_df = TRAUMA_df.rename(columns={'ECODE':'PRIMARYECODEICD10', 'INJTYPE':'TRAUMATYPE','GENDER':'SEX', 'EDDISP':'EDDISCHARGEDISPOSITION', 'HOSPDISP':'HOSPDISCHARGEDISPOSITION', 'HEMORRHAGECTRLMINS':'HMRRHGCTRLSURGMINS', 'HEMORRHAGECTRLDAYS':'HMRRHGCTRLSURGDAYS', 'AGE':'AGEYEARS', 'HEMORRHAGE_CTRL_STYPE_CODE': 'HMRRHGCTRLSURGTYPE', 'SBP':'EMSSBP', 'PULSE':'EMSPULSERATE', 'RR':'EMSRESPIRATORYRATE', 'GCSTOT':'EMSTOTALGCS'})
    [print(i) for i in TRAUMA_df.columns.tolist()]
    TRAUMA_df = TRAUMA_df.loc[:,['SEX', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'TRAUMATYPE', 'MECHANISM']]
    
    # Select only the numeric columns
    numeric_cols = TRAUMA_df.select_dtypes(include=[np.number]).columns

    # Replace negative values with NaN only in numeric columns
    TRAUMA_df[numeric_cols] = TRAUMA_df[numeric_cols].where(TRAUMA_df[numeric_cols] >= 0, np.nan)

    
    TRAUMA_df.to_csv('/Users/JakeCanfield/Documents/Trauma_Surgery_Research/data/Combined_data/appTRAUMA_df_%i.csv'%year, index=False)

