#%%
from warnings import warn

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
import statsmodels.stats.multitest as st


#%%

def eda(condition, subject, fname, num = None, df_rs = None, bad = None):
    
    fname = f'../data/raw/{condition}/{subject}/{fname}'
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    # min lentgh across subjects, to avoid errors
    df = df_data[:293434] #min(326005, 324840, 320150, 313305, 327750, 330335, 295405, 340695, 293435, 329035)
    # length cosnsitent for EDA and HR.
    
    if 'GSR' in df.columns:

        if df_rs is None:
            df_eda, info_eda = nk.eda_process(df['GSR'], sampling_rate=250) #neurokit method
            
            time_x = None
            # print('\n Entre \n')
            # plt.figure(numero)
            # plt.plot(df['time'], df_eda['EDA_Tonic'], label = f'{carpeta}')
            # plt.xlabel('Tiempo (s)')
            # plt.legend()

            
        else:
                    
            df_eda, info_eda = nk.eda_process(df['GSR'], sampling_rate=250) #neurokit method
            df_eda['EDA_Tonic'] = df_eda['EDA_Tonic'] - np.mean(df_rs['EDA_Tonic'])
            
            plt.figure(num)
            plt.plot(df['time'], df_eda['EDA_Tonic'], label = f'{subject}')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.tight_layout()
            
            time_x = df['time']
            
        return df_eda, info_eda, time_x
        
    else:
        bad = [1]
        
        return bad, 0
       
#%%
# Specifying dose (`dosis`) and creating a dataframe with the dose  for each subject
# High dose as 'Alta' and Low dose as 'Baja'

dosis = [['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta']]

columns = ['Dose_Session_1', 'Dose_Session_2'] # definimos los nombres de las columnas
indices = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']

dose = pd.DataFrame(dosis, columns=columns, index = indices)

#%% 

# condition = ['DMT_1', 'Reposo_1', 'DMT_2', 'Reposo_2']
# subjects = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']

## Plot just subject that are available
subjects = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']

#### Subjects with no EDA signal
#### S08 (DMT_2), S10 (DMT_2), S11 (DMT_2), S12 (DMT_2, DMT_1 esta bien igual), S15 (DMT_2)

#%% Iterating over subjects to extract EDA data

plt.close('all')  

promedio_dmt_tonic_alta = []
promedio_dmt_phasic_alta = []
promedio_dmt_tonic_baja = []
promedio_dmt_phasic_baja = []
promedio_tiempo_alto = []
promedio_tiempo_bajo = []

for sub in subjects:
        
    # if dose is low in session 1
    if dose['Dose_Session_1'][sub] == 'Baja':
        
        j = 2
        condition = 'Reposo_1'
        fname = f'{sub}_RS_Session1_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs, none = eda(condition, sub, fname, j + 1)
        
        condition = 'Reposo_2'
        fname = f'{sub}_RS_Session2_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs, none = eda(condition, sub, fname, j)
        
        i = 1
        condition = 'DMT_1'
        fname = f'{sub}_DMT_Session1_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo = eda(condition, sub, fname, i + 1, df_eda_baja_rs)

        condition = 'DMT_2'
        fname = f'{sub}_DMT_Session2_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto = eda(condition, sub, fname, i, df_eda_alta_rs)
        
        
    # if dose is high in session 1
    else:
        
        j = 2
        condition = 'Reposo_1'
        fname = f'{sub}_RS_Session1_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs, none = eda(condition, sub, fname, j)
        
        condition = 'Reposo_2'
        fname = f'{sub}_RS_Session2_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs, none = eda(condition, sub, fname , j + 1)
        
        i = 1
        condition = 'DMT_1'
        fname = f'{sub}_DMT_Session1_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto = eda(condition, sub, fname, i, df_eda_alta_rs)

        condition = 'DMT_2'
        fname = f'{sub}_DMT_Session2_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo = eda(condition, sub, fname, i+1, df_eda_baja_rs)
        
     
    ## Acá agrego el hecho de que guarde los datos de cada uno en listas para después promediar cada punto
    if len(df_eda_alta_dmt) != 1 and len(df_eda_alta_rs) != 1:
        
        prom_dmt_tonic = df_eda_alta_dmt['EDA_Tonic']
        prom_dmt_phasic = df_eda_alta_dmt['EDA_Phasic']
        prom_time = tiempo_alto
        promedio_dmt_tonic_alta.append(prom_dmt_tonic)
        promedio_dmt_phasic_alta.append(prom_dmt_phasic)
        promedio_tiempo_alto.append(prom_time)
        
        
    else:
        print(f'Sujeto {sub} tiene datos (con dosis DMT Alta) de EDA indescifrables')
        
    if len(df_eda_baja_dmt) != 1 and len(df_eda_baja_rs) != 1:
        
        prom_dmt_tonic = df_eda_baja_dmt['EDA_Tonic']
        prom_dmt_phasic = df_eda_baja_dmt['EDA_Phasic']
        prom_time = tiempo_bajo
        promedio_dmt_tonic_baja.append(prom_dmt_tonic)
        promedio_dmt_phasic_baja.append(prom_dmt_phasic)
        promedio_tiempo_bajo.append(prom_time)
        

#%% 
measure = 'EDA_Tonic'

# Lista de datos procesados pasada a dataframe
df_tonic_alta_guardar = pd.DataFrame(promedio_dmt_tonic_alta).T
df_tonic_alta_guardar.columns = subjects

df_tonic_baja_guardar = pd.DataFrame(promedio_dmt_tonic_baja).T
df_tonic_baja_guardar.columns = subjects

#%%

measure = 'EDA_Phasic'

# Lista de datos procesados pasada a dataframe
df_phasic_alta_guardar = pd.DataFrame(promedio_dmt_phasic_alta).T
df_phasic_alta_guardar.columns = subjects

df_phasic_baja_guardar = pd.DataFrame(promedio_dmt_phasic_baja).T
df_phasic_baja_guardar.columns = subjects

#%%
df_time = pd.DataFrame(promedio_tiempo_alto).T
# Obtener la primera columna de df_time
time_column = df_time.iloc[:, 0]

# Agregar la columna de tiempo a cada dataframe
df_tonic_alta_guardar.insert(0, 'Time', time_column)
df_tonic_baja_guardar.insert(0, 'Time', time_column)
df_phasic_alta_guardar.insert(0, 'Time', time_column)
df_phasic_baja_guardar.insert(0, 'Time', time_column)

# Guardar los dataframes como CSV
df_tonic_alta_guardar.to_csv(f'./X/EDA_Tonic_DMT_high.csv', index = False)
df_tonic_baja_guardar.to_csv(f'./X/EDA_Tonic_DMT_low.csv', index = False)
df_phasic_alta_guardar.to_csv(f'./X/EDA_Phasic_DMT_high.csv', index = False)
df_phasic_baja_guardar.to_csv(f'./X/EDA_Phasic_DMT_low.csv', index = False)


# %%
# Downsampling the data to 0.25 Hz=
from scipy.signal import decimate

# Definir la nueva frecuencia de muestreo
new_sampling_rate = 0.25  # Hz
downsampling_factor = int(250 / new_sampling_rate)

def downsample_data(df, downsampling_factor):
    # Downsampleando los datos
    downsampled_data = {}
    num_samples = len(df) // downsampling_factor  # Número de muestras después del downsampling
    downsampled_data['Time'] = np.arange(num_samples) * (1 / new_sampling_rate)

    for col in df.columns[1:]:
        downsampled_data[col] = decimate(df[col], downsampling_factor, ftype='iir')[:num_samples]

    return pd.DataFrame(downsampled_data)

# Downsampleando los dataframes y ajustando la columna de tiempo
df_tonic_alta_downsampled = downsample_data(df_tonic_alta_guardar, downsampling_factor)
df_tonic_baja_downsampled = downsample_data(df_tonic_baja_guardar, downsampling_factor)
df_phasic_alta_downsampled = downsample_data(df_phasic_alta_guardar, downsampling_factor)
df_phasic_baja_downsampled = downsample_data(df_phasic_baja_guardar, downsampling_factor)

# Guardar los dataframes como CSV
df_tonic_alta_downsampled.to_csv(f'./X/EDA_Tonic_DMT_high_downsampled.csv', index=False)
df_tonic_baja_downsampled.to_csv(f'./X/EDA_Tonic_DMT_low_downsampled.csv', index=False)
df_phasic_alta_downsampled.to_csv(f'./X/EDA_Phasic_DMT_high_downsampled.csv', index=False)
df_phasic_baja_downsampled.to_csv(f'./X/EDA_Phasic_DMT_low_downsampled.csv', index=False)

# %%
