
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
#phys_annotations_glhmm = glhmm.glhmm(model_beta='no', K=3, covtype='full')
phys_annotations_glhmm = glhmm.glhmm(model_beta='state', K=3, covtype='full')

#%%
# Check the hyperparameters of the object to make sure the model is defined as we planned:
print(phys_annotations_glhmm.hyperparameters)

#%%
# Train the model
gamma_values,_,_ = phys_annotations_glhmm.train(X=None,
                             Y=EDA_data, 
                             indices=T_t)

#%%

# Inspect the model
K = phys_annotations_glhmm.hyperparameters["K"] # the number of states
q = EDA_data.shape[1] # the number of parcels/channels
state_betas = np.zeros(shape=(2,q,K))
state_betas = phys_annotations_glhmm.get_betas()

#%%

# Since we here defined to be time-varying, i.e., state-dependent,
# we have a matrix describing the interaction between each of the 2 EDA components (Phasic and Tonic)
# and the Emotional Intensity annotations for each of the 4 states:

cmap = "coolwarm"
ytick =["EDA Phasic", "EDA Tonic"]
for k in range(K):
    plt.subplot(2,2,k+1)
    plt.imshow(state_betas[:,:,k], cmap=cmap,aspect='auto', interpolation='none')
    plt.colorbar()
    plt.ylabel('Emotional Intensity (Annotation)')
    plt.yticks(np.arange(2), ytick)
    plt.xlabel('')
    plt.title(f"Betas for state #{k+1}")
plt.subplots_adjust(hspace=0.5, wspace=1)
plt.show()

#%%
# State means and covariances
state_means = np.zeros(shape=(q, K))
for k in range(K):
    state_means[:,k] = phys_annotations_glhmm.get_mean(k) # the state means in the shape (no. features, no. states)
state_FC = np.zeros(shape=(q, q, K))
for k in range(K):
    state_FC[:,:,k] = phys_annotations_glhmm.get_covariance_matrix(k=k) # the state covariance matrices in the shape (no. features, no. features, no. states)

# Plot them
plt.imshow(state_means, cmap=cmap, interpolation="none")
plt.colorbar(label='Activation Level')  # Label for color bar
plt.title("State Mean Activation")
plt.xticks(np.arange(K), np.arange(1, K + 1))

# Etiquetas específicas para el eje Y
plt.gca().set_xlabel('State')
plt.gca().set_ylabel('Physiology')
plt.yticks(np.arange(q), ['EDA Phasic', 'EDA Tonic'])  # Etiquetas en el eje Y

plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# %%

for k in range(K):
    plt.subplot(2,2,k+1)
    plt.imshow(state_FC[:,:,k], cmap=cmap, interpolation="none")
    plt.xlabel('EDA Features')
    plt.ylabel('EDA Features')
    plt.colorbar()
    plt.title("State covariance\nstate #%s" % (k+1))
plt.subplots_adjust(hspace=0.7, wspace=0.8)
plt.show()

# %%
# Dynamics: Transition probabilities and Viterbi path

TP = phys_annotations_glhmm.P.copy() # the transition probability matrix

# Plot Transition Probabilities
plt.figure(figsize=(7, 4))

# Plot 1: Original Transition Probabilities
plt.subplot(1, 2, 1)
plt.imshow(TP, cmap=cmap, interpolation='nearest')  # Improved color mapping
plt.title('Transition Probabilities')
plt.xlabel('To State')
plt.ylabel('From State')
plt.colorbar(fraction=0.046, pad=0.04)

# Plot 2: Transition Probabilities without Self-Transitions
TP_noself = TP - np.diag(np.diag(TP))  # Remove self-transitions
TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)  # Normalize probabilities
plt.subplot(1, 2, 2)
plt.imshow(TP_noself2, cmap=cmap, interpolation='nearest')  # Improved color mapping
plt.title('Transition Probabilities\nwithout Self-Transitions')
plt.xlabel('To State')
plt.ylabel('From State')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()  # Adjust layout for better spacing

plt.show()

#%%
# Viterbi path: the most likely sequence of states given the data
vpath = phys_annotations_glhmm.decode(X=emotional_intensity_data, Y=EDA_data, indices=T_t, viterbi=True)

#%%
graphics.plot_vpath(vpath, title="Viterbi path")

# %%
num_subject = 5

graphics.plot_vpath(vpath[T_t[num_subject,0]:T_t[num_subject,1],:], title="Viterbi path")

# %%
import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np

# Usar un backend sin interfaz gráfica para evitar mostrar la figura
matplotlib.use('Agg')

# Asegúrate de que el directorio ./plots/viterbi exista
os.makedirs('./plots/viterbi_X_Y', exist_ok=True)

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
    plt.savefig(f"./plots/viterbi/vpath_participant_{num_subject}_tonic.png", bbox_inches='tight')

    # Cerrar la figura para liberar memoria
    plt.close()

    # Graficar el Viterbi path con la señal de EDA Phasic sobreimpuesta
    graphics.plot_vpath(vpath_subject, signal=eda_phasic_normalized,
                        title=f"Viterbi Path with Normalized EDA Phasic Signal - Participant {num_subject}",
                        signal_label="EDA Phasic (Normalized)", time_conversion_rate=250)
    # Guardar la figura
    plt.savefig(f"./plots/viterbi/vpath_participant_{num_subject}_phasic.png", bbox_inches='tight')

    # Cerrar la figura para liberar memoria
    plt.close()



# %%
# Across-Sessions Within Subject Testing with glhmm toolbox
