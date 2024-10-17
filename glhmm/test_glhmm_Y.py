
#%%
###################################
############## GLHMM ##############
###################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics

emotional_intensity_data = pd.read_csv('./y/data.csv', header=None).to_numpy()
EDA_data = pd.read_csv('./X/dataX.csv', header=None).to_numpy()
T_t = pd.read_csv('./X/T.csv', header=None).to_numpy()

#%%
emotional_intensity_data,_ = preproc.preprocess_data(emotional_intensity_data, T_t, standardise = True)
EDA_data,_ = preproc.preprocess_data(EDA_data, T_t, standardise = True)

#%%
# Initialise and train a GLHMM
phys_annotations_glhmm = glhmm.glhmm(model_beta='no', K=3, covtype='full')
#phys_annotations_glhmm = glhmm.glhmm(model_beta='state', K=3, covtype='full')

#%%
# Check the hyperparameters of the object to make sure the model is defined as we planned:
print(phys_annotations_glhmm.hyperparameters)

#%%
# Train the model
gamma_values,_,_ = phys_annotations_glhmm.train(X=None,
                             Y=EDA_data, 
                             indices=T_t)

#%%

K = phys_annotations_glhmm.hyperparameters["K"] # the number of states
q = EDA_data.shape[1] # the number of parcels/channels
state_means = np.zeros(shape=(q, K))
state_means = phys_annotations_glhmm.get_means() # the state means in the shape (no. features, no. states)

#%%

cmap = "coolwarm"
ytick = ["EDA Phasic", "EDA Tonic"]

plt.imshow(state_means, cmap=cmap, interpolation="none")
plt.colorbar(label='Activation Level')  # Label for color bar
plt.title("State mean activation")
plt.xticks(np.arange(K), np.arange(1, K+1))
plt.gca().set_xlabel('State')
plt.gca().set_ylabel('EDA Features')
plt.yticks(np.arange(2), ytick)  # Configurar los ticks del eje y con tus etiquetas
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

#%%
# State means and covariances
state_FC = np.zeros(shape=(q, q, K))
for k in range(K):
    state_FC[:,:,k] = phys_annotations_glhmm.get_covariance_matrix(k=k) # the state covariance matrices in the shape (no. features, no. features, no. states)
# %%

cmap = "coolwarm"
ytick = ["EDA Phasic", "EDA Tonic"]

# Graficar las matrices de covarianza para cada estado
for k in range(K):
    plt.subplot(2,2,k+1)
    plt.imshow(state_FC[:,:,k], cmap=cmap, interpolation="none")
    plt.xlabel('EDA Features')
    plt.ylabel('EDA Features')
    plt.colorbar()
    plt.title(f"State covariance\nstate #{k+1}")
    
    # Ajustar los ticks del eje y y el eje x con tus etiquetas personalizadas
    plt.yticks(np.arange(2), ytick)
    plt.xticks(np.arange(2), ytick)

plt.subplots_adjust(hspace=0.7, wspace=0.8)
plt.show()

#%%
%matplotlib qt
# Transition probabilities
TP = phys_annotations_glhmm.P.copy()  # the transition probability matrix

# Plot Transition Probabilities
plt.figure(figsize=(7, 4))

# Definir etiquetas de los estados
state_labels = np.arange(1, TP.shape[0] + 1)

# Plot 1: Original Transition Probabilities
plt.subplot(1, 2, 1)
plt.imshow(TP, cmap=cmap, interpolation='nearest')  # Improved color mapping
plt.title('Transition Probabilities')
plt.xlabel('To State')
plt.ylabel('From State')
plt.colorbar(fraction=0.046, pad=0.04)

# Agregar etiquetas personalizadas en los ejes X e Y
plt.xticks(np.arange(TP.shape[0]), state_labels)
plt.yticks(np.arange(TP.shape[0]), state_labels)

# Plot 2: Transition Probabilities without Self-Transitions
TP_noself = TP - np.diag(np.diag(TP))  # Remove self-transitions
TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)  # Normalize probabilities
plt.subplot(1, 2, 2)
plt.imshow(TP_noself2, cmap=cmap, interpolation='nearest')  # Improved color mapping
plt.title('Transition Probabilities\nwithout Self-Transitions')
plt.xlabel('To State')
plt.ylabel('From State')
plt.colorbar(fraction=0.046, pad=0.04)

# Agregar etiquetas personalizadas en los ejes X e Y
plt.xticks(np.arange(TP_noself2.shape[0]), state_labels)
plt.yticks(np.arange(TP_noself2.shape[0]), state_labels)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()


#%%
# Viterbi path

vpath = phys_annotations_glhmm.decode(X=None, Y=EDA_data, indices=T_t, viterbi=True)

graphics.plot_vpath(vpath, title="Viterbi path")

#%%

num_subject = 10
graphics.plot_vpath(vpath[T_t[num_subject,0]:T_t[num_subject,1],:], title="Viterbi path")

#%%

import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np

# Usar un backend sin interfaz gráfica para evitar mostrar la figura
matplotlib.use('Agg')

# Asegúrate de que el directorio ./plots/viterbi exista
os.makedirs('./plots/viterbi_Y', exist_ok=True)

# Definir el rango de participantes
participant_range = range(11)  # De 0 a 10

for num_subject in participant_range:
    # Extraer el Viterbi path para el sujeto seleccionado
    vpath_subject = vpath[T_t[num_subject, 0]:T_t[num_subject, 1], :]

    # Extraer la señal de EDA_Tonic y EDA_Phasic para el sujeto seleccionado
    eda_tonic_subject = EDA_data[T_t[num_subject, 0]:T_t[num_subject, 1], 1]
    eda_phasic_subject = EDA_data[T_t[num_subject, 0]:T_t[num_subject, 1], 0]

    # Normalizar las señales de EDA_Tonic y EDA_Phasic
    eda_tonic_normalized = (eda_tonic_subject - np.min(eda_tonic_subject)) / (np.max(eda_tonic_subject) - np.min(eda_tonic_subject))
    eda_phasic_normalized = (eda_phasic_subject - np.min(eda_phasic_subject)) / (np.max(eda_phasic_subject) - np.min(eda_phasic_subject))

    # Graficar el Viterbi path con la señal de EDA Tonic sobreimpuesta
    graphics.plot_vpath(vpath_subject, signal=eda_tonic_normalized,
                        title=f"Viterbi Path with Normalized EDA Tonic Signal - Participant {num_subject}",
                        signal_label="EDA Tonic (Normalized)", time_conversion_rate=250)
    # Guardar la figura
    plt.savefig(f"./plots/viterbi_Y/vpath_participant_{num_subject}_tonic.png", bbox_inches='tight')

    # Cerrar la figura para liberar memoria
    plt.close()

    # Graficar el Viterbi path con la señal de EDA Phasic sobreimpuesta
    graphics.plot_vpath(vpath_subject, signal=eda_phasic_normalized,
                        title=f"Viterbi Path with Normalized EDA Phasic Signal - Participant {num_subject}",
                        signal_label="EDA Phasic (Normalized)", time_conversion_rate=250)
    # Guardar la figura
    plt.savefig(f"./plots/viterbi_Y/vpath_participant_{num_subject}_phasic.png", bbox_inches='tight')

    # Cerrar la figura para liberar memoria
    plt.close()
# %%
