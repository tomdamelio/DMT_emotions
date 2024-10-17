import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Statistics
from scipy import stats
import statsmodels.stats.multitest as st

#%% 

# Define el directorio que quieres leer
directorio = "../data/derivatives/annotations/resampled"

# Obtiene la lista de archivos en el directorio y los ordena alfabeticamente
archivos = sorted(os.listdir(directorio))

# Filtra solo los archivos
archivos_ordenados = [archivo for archivo in archivos if os.path.isfile(os.path.join(directorio, archivo))]

#%% Tabla de dosis

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
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta']]

columnas = ['DMT_1', 'DMT_2'] # definimos los nombres de las columnas
indices = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S13','S15','S16','S17','S18', 'S19','S20']

dosis = pd.DataFrame(dosis, columns=columnas, index=indices)

#%%
# Create a list of dataframes for each condition

alta = []
baja = []
rs_alta = []
rs_baja = [] 

# Create a function that concatenates the dataframes of each subject 

def concatenacion(fname):
    
    mat = scipy.io.loadmat(f'../data/derivatives/annotations/resampled/{fname}')

    dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

    df_dibujo = pd.DataFrame(mat['dimensions'], columns=dimensiones)
    
    carpeta = fname[0:3]
    carpeta = carpeta[0].upper() + carpeta[1:]  # Convert first letter to uppercase
    
    # Create the time column
    num_samples = df_dibujo.shape[0]
    time = np.arange(0, num_samples * 4, 4)  # Correct time column for 0.25 Hz sampling rate
    
    df_dibujo.insert(0, 'Time', time)  # Insert the time column at the first position
    df_dibujo.insert(0, 'Participant', carpeta)  # Insert the participant column at the first position
    
    experimento = fname[4:6]
    
    # Define DMT_1, DMT_2, RS_1 y RS_2
    if experimento == 'DM':
        experimento = experimento + fname[6:7] + '_' + fname[15:16]
    else:
        experimento = experimento + '_' + fname[14:15]

    # If condition is "low" dose of DMT
    if (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis.loc[carpeta, experimento] == 'Baja': 
        baja.append(df_dibujo)
    
    # If condition is "high" dose of DMT        
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis.loc[carpeta, experimento] == 'Alta':
        alta.append(df_dibujo)

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre acá
    elif (experimento == 'RS_1' or experimento == 'RS_2') and dosis.loc[carpeta, 'DMT_' + experimento[3]] == 'Alta':
        rs_alta.append(df_dibujo)
        
    else:
        rs_baja.append(df_dibujo)

#%% Corro la funcion anterior

for archivo in archivos:
    concatenacion(archivo)

# Concatenar los dataframes individuales en uno solo por cada condición
df_alta = pd.concat(alta, ignore_index=True)
df_baja = pd.concat(baja, ignore_index=True)
df_rs_alta = pd.concat(rs_alta, ignore_index=True)
df_rs_baja = pd.concat(rs_baja, ignore_index=True)

#%%
# Guardar los dataframes en CSV
df_alta.to_csv('./y/all_dimensions_DMT_high.csv', index=False)
df_baja.to_csv('./y/all_dimensions_DMT_low.csv', index=False)
df_rs_alta.to_csv('./y/all_dimensions_RS_high.csv', index=False)
df_rs_baja.to_csv('./y/all_dimensions_RS_low.csv', index=False)

# Extraer Emotional_Intensity y guardarlo en archivos CSV
def extraer_emotional_intensity(df, filename, participantes):
    # Filtrar los participantes específicos
    df_filtered = df[df['Participant'].isin(participantes)]
    df_emotional = df_filtered.pivot(index='Time', columns='Participant', values='Emotional_Intensity')
    df_emotional.reset_index(inplace=True)
    df_emotional.to_csv(filename, index=False)

# Lista de participantes deseados
participantes_deseados = ['S04', 'S05', 'S06', 'S07', 'S09', 'S13', 'S16', 'S17', 'S18', 'S19', 'S20']

extraer_emotional_intensity(df_alta, './y/emotional_intensity_DMT_high.csv', participantes_deseados)
extraer_emotional_intensity(df_baja, './y/emotional_intensity_DMT_low.csv', participantes_deseados)
extraer_emotional_intensity(df_rs_alta, './y/emotional_intensity_RS_high.csv', participantes_deseados)
extraer_emotional_intensity(df_rs_baja, './y/emotional_intensity_RS_low.csv', participantes_deseados)


#%%
from scipy.signal import resample

# Definir la frecuencia de muestreo original y la nueva
original_sampling_rate = 0.25  # Hz
new_sampling_rate = 250  # Hz
upsampling_factor = int(new_sampling_rate / original_sampling_rate)

def upsample_data(df, upsampling_factor):
    # Resamplear los datos
    num_samples = len(df) * upsampling_factor  # Número de muestras después del upsampling
    upsampled_data = {}
    upsampled_data['Time'] = np.arange(num_samples) * (1 / new_sampling_rate)

    for col in df.columns[1:]:
        upsampled_data[col] = resample(df[col], num_samples)

    return pd.DataFrame(upsampled_data)

# Upsampleando los dataframes y ajustando la columna de tiempo
df_emotional_alta_upsampled = upsample_data(df_emotional_alta, upsampling_factor)
df_emotional_baja_upsampled = upsample_data(df_emotional_baja, upsampling_factor)
df_emotional_rs_alta_upsampled = upsample_data(df_emotional_rs_alta, upsampling_factor)
df_emotional_rs_baja_upsampled = upsample_data(df_emotional_rs_baja, upsampling_factor)

# Guardar los dataframes como CSV
df_emotional_alta_upsampled.to_csv('./y/emotional_intensity_DMT_high_upsampled.csv', index=False)
df_emotional_baja_upsampled.to_csv('./y/emotional_intensity_DMT_low_upsampled.csv', index=False)
df_emotional_rs_alta_upsampled.to_csv('./y/emotional_intensity_RS_high_upsampled.csv', index=False)
df_emotional_rs_baja_upsampled.to_csv('./y/emotional_intensity_RS_low_upsampled.csv', index=False)



#%%

from scipy.signal import resample

# Definir la frecuencia de muestreo original y la nueva
original_sampling_rate = 0.25  # Hz
new_sampling_rate = 250  # Hz
upsampling_factor = int(new_sampling_rate / original_sampling_rate)

def upsample_data(df, upsampling_factor):
    # Resamplear los datos
    num_samples = len(df) * upsampling_factor  # Número de muestras después del upsampling
    upsampled_data = {}
    upsampled_data['Time'] = np.arange(num_samples) * (1 / new_sampling_rate)

    for col in df.columns[1:]:
        upsampled_data[col] = resample(df[col], num_samples)

    return pd.DataFrame(upsampled_data)


# Cargar los archivos guardados y aplicar upsampling
df_emotional_alta = pd.read_csv('./y/emotional_intensity_DMT_high.csv')
df_emotional_baja = pd.read_csv('./y/emotional_intensity_DMT_low.csv')
df_emotional_rs_alta = pd.read_csv('./y/emotional_intensity_RS_high.csv')
df_emotional_rs_baja = pd.read_csv('./y/emotional_intensity_RS_low.csv')

# Aplicar upsampling
df_emotional_alta_upsampled = upsample_data(df_emotional_alta, upsampling_factor)
df_emotional_baja_upsampled = upsample_data(df_emotional_baja, upsampling_factor)
df_emotional_rs_alta_upsampled = upsample_data(df_emotional_rs_alta, upsampling_factor)
df_emotional_rs_baja_upsampled = upsample_data(df_emotional_rs_baja, upsampling_factor)

# Guardar los dataframes como CSV
df_emotional_alta_upsampled.to_csv('./y/emotional_intensity_DMT_high_upsampled.csv', index=False)
df_emotional_baja_upsampled.to_csv('./y/emotional_intensity_DMT_low_upsampled.csv', index=False)
df_emotional_rs_alta_upsampled.to_csv('./y/emotional_intensity_RS_high_upsampled.csv', index=False)
df_emotional_rs_baja_upsampled.to_csv('./y/emotional_intensity_RS_low_upsampled.csv', index=False)

#%%
###### SOLO CORRER HASTA ACA. EL RESTO DE PCA NO LO IMPLENTE AUN ######











#%%
# Esta bien si tengo 300*18 en alta, 300*19 en baja, 150*19 y 150*19
    
todos_dfs = [alta, baja, rs_alta, rs_baja]

    
#%% Concateno ahora y corro el pca con fit transform

df_concatenados = pd.concat(todos_dfs, ignore_index = True)

indices = ['PC1','PC2']
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']


# Componentes a quedarme, importante aclararlo
cantidad_pc = 2 # son 15, si resto por 12 me quedo con las primeras 3 componentes

# Creo un pipeline donde le hace el standard Scaler (que pone cada columna independiente a un z score)
# y después hace el pca, le aclaro que me voy a quedar con la cantidad de PC aclarado
X = scale(df_concatenados)
pca = PCA(n_components = cantidad_pc)
X = pca.fit_transform(X)

df_pca = pd.DataFrame(data = X, columns = indices)


loadings = pd.DataFrame(pca.components_.T, columns = indices, index = dimensiones)
top_3_pc1 = loadings['PC1'].abs().nlargest(3)
top_3_pc2 = loadings['PC2'].abs().nlargest(3)

print('PC1:\n', top_3_pc1)
print()
print('PC2:\n', top_3_pc2)
print()
print('PC1 explained variance ratio:', round(pca.explained_variance_ratio_[0],3)*100, '%')
print('PC2 explained variance ratio:', round(pca.explained_variance_ratio_[1],3)*100, '%')


#%% Separo los dataframes en condiciones

sujetos = 18 # el s12 no tiene un archivo asi que di de baja todos los suyos

df_alta = df_pca[:300*sujetos].reset_index(drop = True)
df_baja = df_pca[300*sujetos:300*sujetos + 300*sujetos].reset_index(drop = True)
df_rs_alta = df_pca[300*sujetos + 300*sujetos : 300*sujetos + 300*sujetos + 150*sujetos].reset_index(drop = True)
df_rs_baja = df_pca[300*sujetos + 300*sujetos + 150*sujetos : 300*sujetos + 300*sujetos + 150*sujetos + 150*sujetos].reset_index(drop = True)

dfs = [df_alta,df_baja,df_rs_alta,df_rs_baja]

# %%
