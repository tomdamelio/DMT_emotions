#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics

#%%
###################################
######### X preprocessing #########
###################################

# Paths to the CSV files
path_phasic = './X/EDA_Phasic_DMT_high.csv'
path_tonic = './X/EDA_Tonic_DMT_high.csv'

# Reading the CSV files
eda_phasic = pd.read_csv(path_phasic)
eda_tonic = pd.read_csv(path_tonic)

# Converting the data into long format
phasic_long = eda_phasic.melt(id_vars=["Time"], var_name="Subject", value_name="EDA_Phasic")
tonic_long = eda_tonic.melt(id_vars=["Time"], var_name="Subject", value_name="EDA_Tonic")

# Merging the data on Time and Subject
data = pd.merge(phasic_long, tonic_long, on=["Time", "Subject"])

#%%
# Eliminando las columnas 'Time' y 'Subject'
data = data.drop(columns=['Time', 'Subject'])

# Guardar los datos transformados en un nuevo archivo CSV (opcional)
data.to_csv('./X/dataX.csv', index=False, header=False)

# %%
# Inicializar las listas para los índices de inicio y fin
start_indices = []
end_indices = []

# Valor inicial y tamaño del incremento
initial_value = 0
increment = 293434

# Generar los índices
current_value = initial_value
while current_value < 3227774:
    start_indices.append(current_value)
    current_value += increment
    end_indices.append(current_value)

# Crear el DataFrame T
T = pd.DataFrame({'Start': start_indices, 'End': end_indices})

# Guardar T en un nuevo archivo CSV sin encabezado ni índice
T.to_csv('./X/T.csv', header=False, index=False)


# %%
###################################
######### y preprocessing #########
###################################


# Path to the CSV file
path_emotional_intensity = './y/emotional_intensity_DMT_high_upsampled.csv'

# Reading the CSV file
emotional_intensity = pd.read_csv(path_emotional_intensity)

# Filtering the data to include only rows where Time <= 1168
filtered_emotional_intensity = emotional_intensity[emotional_intensity['Time'] <= 1173.732]

# Converting the data into long format
emotional_long = filtered_emotional_intensity.melt(id_vars=["Time"], var_name="Subject", value_name="Emotional_intensity")

# Eliminando las columnas 'Time' y 'Subject'
emotional_long = emotional_long.drop(columns=['Time', 'Subject'])

# Guardar los datos transformados en un nuevo archivo CSV sin nombres de columnas
emotional_long.to_csv('./y/data.csv', index=False, header=False)

# Visualización del DataFrame resultante (opcional)
print(emotional_long.head())

