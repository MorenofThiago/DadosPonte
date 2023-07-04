# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:25:35 2023

@author: Thiago Moreno Fernandes

Algoritmo: Autoencoder para a deteccao de danos com base em sinais 
de aceleracao unidimensional

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import h5py
import glob
import os
import pandas as pd
import scipy.io


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.io import loadmat


#Importa os dados

#Inserir o posicionamento do sensor (VG: Vagão, TF: Truque frontal, TT: Truque Traseiro, RF: Roda frontal)
#Inserir o vagão monitorado (PrimVag: Primeiro vagão, UltVag: Ultimo Vagão)
PosSensor = 'RF'
Vagao = 'PrimVag'

DadosAll = loadmat(f'Data22-06_{PosSensor}_{Vagao}_Cut.mat')   # Todos os conjuntos de dados (Baseline, 5P, 10P, 20P, 50P)

DadosAll.keys()
sorted(DadosAll.keys())
Baseline = DadosAll[f'Acel_{PosSensor}_Baseline{Vagao}']               # Sem dano
Teste_CincoP =  DadosAll[f'Acel_{PosSensor}_CincoP{Vagao}']            # 5% de dano
Teste_DezP =  DadosAll[f'Acel_{PosSensor}_DezP{Vagao}']                # 10% de dano
Teste_VinteP =  DadosAll[f'Acel_{PosSensor}_VinteP{Vagao}']            # 20% de dano
Teste_CinquentaP =  DadosAll[f'Acel_{PosSensor}_CinquentaP{Vagao}']    # 50% de dano


#Divide os dados baseline em dois conjuntos - para treinamento e validacao
baseline_train, baseline_valid = train_test_split(
    Baseline, test_size=0.2, random_state=21
)


#Normalize os dados para [0,1]
min_val = tf.reduce_min(baseline_train)
max_val = tf.reduce_max(baseline_train)

baseline_train = (baseline_train - min_val) / (max_val - min_val)
baseline_valid = (baseline_valid - min_val) / (max_val - min_val)

baseline_train = tf.cast(baseline_train, tf.float32)
baseline_valid = tf.cast(baseline_valid, tf.float32)


#Construir o modelo
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(5830, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

#Observe que o autoencoder é treinado usando apenas os dados do cenario sem dano
history = autoencoder.fit(baseline_train, baseline_train, 
          epochs=200, 
          batch_size=32,
          validation_data=(baseline_valid, baseline_valid),
          shuffle=True)

plt.plot(history.history["loss"], label="MSE Treinamento")
plt.plot(history.history["val_loss"], label="MSE Validação")
plt.legend()
plt.show()

encoded_data = autoencoder.encoder(baseline_train).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


fig = plt.figure()
ax1 = fig.add_subplot(111)

fig.suptitle('Reconstrução do sinal do cenário sem dano')
plt.plot(baseline_train[0], 'b', linewidth=0.7)
plt.plot(decoded_data[0], 'r', linewidth=0.7)
plt.fill_between(np.arange(5830), decoded_data[0], baseline_train[0], color='lightcoral')
plt.legend(loc='lower right', labels=["Entrada", "Reconstrução", "Erro de reconstrução"])
ax1.margins(x=0)
plt.ylabel("Aceleração (m/s²)")
plt.xlabel("Posição (cm)")
plt.savefig(f'Reconstrucao_{PosSensor}_SemDano.png', dpi=300)
plt.show()


############### Deteccao de danos dos dados de teste ##############
ntestes = 4
Teste = np.zeros((ntestes, Teste_CincoP.shape[0], Teste_CincoP.shape[1]))
yte = np.zeros((ntestes,Teste_CincoP.shape[0]))
Media = np.zeros((ntestes))
Desvio = np.zeros((ntestes))
#teste_data = np.zeros((ntestes, Teste_CincoP.shape[0], Teste_CincoP.shape[1]))

Caso = ['Teste_CincoP','Teste_DezP','Teste_VinteP','Teste_CinquentaP']

for t in range(ntestes):
    
    Teste_Caso = Caso[t]
    
    match Teste_Caso:
        case 'Teste_CincoP':
             TesteName = '5%'
        case 'Teste_DezP':
             TesteName = '10%'
        case 'Teste_VinteP':
             TesteName = '20%'
        case 'Teste_CinquentaP':
            TesteName = '50%'
        case _:
            TesteName = '0%'
    
    Teste[t] = eval(Teste_Caso)
    
    teste_data = (Teste[t,:] - min_val) / (max_val - min_val)
    teste_data = tf.cast(teste_data, tf.float32)


    #Crie um gráfico semelhante, desta vez para o teste com dano
    encoded_data = autoencoder.encoder(teste_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.suptitle(f'Reconstrução do sinal do cenário com dano de {TesteName}')
    plt.plot(teste_data[0], 'b', linewidth=0.7)
    plt.plot(decoded_data[0], 'r', linewidth=0.7)
    plt.fill_between(np.arange(5830), decoded_data[0], teste_data[0], color='lightcoral')
    plt.legend(loc='lower right', labels=["Entrada", "Reconstrução", "Erro de reconstrução"])
    plt.ylabel("Aceleração (m/s²)")
    plt.xlabel("Posição (cm)")
    ax1.margins(x=0)
    plt.savefig(f'Reconstrucao_{PosSensor}_{TesteName}.png', dpi=300)
    plt.show()
    
    ## Danos
    #Detecção do dano pelo calculo da perda de reconstrução maior que um limite 
    #fixo
    reconstructions = autoencoder.predict(baseline_train)
    train_loss = tf.keras.losses.mae(reconstructions, baseline_train)
    
    plt.hist(train_loss[None,:], bins=25)
    plt.xlabel("MSE Treinamento")
    plt.ylabel("Nº de amostras")
    plt.show()
    
    #Escolha um valor limite que seja um desvio padrão acima da média.
    threshold_train = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold_train)
    
    
    #Se você examinar o erro de reconstrução dos exemplos anômalos no conjunto de 
    #teste, notará que a maioria tem um erro de reconstrução maior do que o limite.
    #Variando o limite, você pode ajustar a precisão e a recuperação do seu classificador.
    reconstructions = autoencoder.predict(teste_data)
    test_loss = tf.keras.losses.mae(reconstructions, teste_data)
    threshold_test = np.mean(test_loss) + np.std(test_loss)
    
    
    plt.hist(test_loss[None, :], bins=25)
    plt.xlabel(f"MSE Teste para o dano de {TesteName}")
    plt.ylabel("Nº de amostras")
    plt.show()
     
    ## Scatter plot
    #def plot_mae (ax1, train_loss, test_loss)
    yte[t,:] = np.array(test_loss)
    Media[t] = np.mean(test_loss)
    Desvio[t] = np.std(test_loss)
    

    
x = np. linspace (0, 4.8, 480)
ytr = np.array(train_loss)

fig = plt.figure()
fig.suptitle('MAE para todos os cenários')
ax1 = fig.add_subplot(111)

ax1.scatter(x[:80], ytr[None, :], s=5, marker="*",  c='black', linewidths=0.5)
ax1.axhline(y=np.mean(train_loss), xmin=0, xmax = (80/480), color='black', linestyle='--', label='Média')
ax1.axhline(y=threshold_train, xmin=0, xmax = (80/480), color='black', linestyle='-', linewidth=0.5)
ax1.axhline(y= (np.mean(train_loss)-np.std(train_loss)), xmin=0, xmax = (80/480), color='black', linestyle='-', linewidth=0.5)
ax1.fill_between(np.linspace(0,0.8,100), (np.mean(train_loss)-np.std(train_loss)), threshold_train, color='black', alpha=.3, label='Média ± desvio-padrão')

ax1.scatter(x[80:180], yte[0, :], s=5, facecolors='none',  marker="v", edgecolors='black', linewidths=0.5)
ax1.axhline(y=np.mean(yte[0, :]), xmin=(80/480), xmax = (180/480), color='gray', linestyle='--')
ax1.axhline(y=Media[0]+Desvio[0], xmin=(80/480), xmax = 180/480, color='gray', linestyle='-', linewidth=0.5)
ax1.fill_between(np.linspace(0.8,1.8,100), Media[0]-Desvio[0], Media[0]+Desvio[0], color='gray', alpha=.3)

ax1.scatter(x[180:280], yte[1, :], s=5, facecolors='none', marker="s", edgecolors='black', linewidths=0.5)
ax1.axhline(y=np.mean(yte[1, :]), xmin=(180/480), xmax = (280/480), color='gray', linestyle='--')
ax1.axhline(y=Media[1]+Desvio[1], xmin=(180/480), xmax = 280/480, color='gray', linestyle='-', linewidth=0.5)
ax1.fill_between(np.linspace(1.8,2.8,100), Media[1]-Desvio[1], Media[1]+Desvio[1], color='gray', alpha=.3)

ax1.scatter(x[280:380], yte[2, :], s=5, facecolors='none', marker="d", edgecolors='black', linewidths=0.5)
ax1.axhline(y=np.mean(yte[2, :]), xmin=(280/480), xmax = (380/480), color='gray', linestyle='--')
ax1.axhline(y=Media[2]+Desvio[2], xmin=(280/480), xmax = (380/480), color='gray', linestyle='-', linewidth=0.5)
ax1.fill_between(np.linspace(2.8,3.8,100), Media[2]-Desvio[2], Media[2]+Desvio[2], color='gray', alpha=.3)

ax1.scatter(x[380:480], yte[3, :], s=5, facecolors='none', marker="o", edgecolors='black', linewidths=0.5)
ax1.axhline(y=np.mean(yte[3, :]), xmin=(380/480), xmax = x[-1], color='gray', linestyle='--')
ax1.axhline(y=Media[3]+Desvio[3], xmin=(380/480), xmax = x[-1], color='gray', linestyle='-', linewidth=0.5)
ax1.fill_between(np.linspace(3.8,4.8,100), Media[3]-Desvio[3], Media[3]+Desvio[3], color='gray', alpha=.3)

ax1.set_ylim([0, 0.05]) #y axis limits
ax1.margins(x=0)

#Linha vertical para dividir dois cenarios
ax1.axvline(x = x[80], color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x = x[180], color='black', linestyle='-', linewidth=0.4)
ax1.axvline(x = x[280], color='black', linestyle='-', linewidth=0.4)
ax1.axvline(x = x[380], color='black', linestyle='-', linewidth=0.4)
plt.scatter
plt.legend(loc='upper right', fontsize=10)
plt.ylabel('MAE')


x1 = [x[40],x[130],x[230],x[330],x[430]]
squad = ['Sem dano','Dano 5%', 'Dano 10%','Dano 20%','Dano 50%']

ax1.set_xticks(x1)
ax1.set_xticklabels(squad, minor=False)
plt.xlabel("Cenário de dano")


plt.show()
fig.savefig(f'MAE_Cut_{PosSensor}_{Vagao}.png', dpi=300)

#Salva os dados do MAE de treinamento e de teste em um arquivo .mat
scipy.io.savemat('Results-MAEtrain22-06.mat', {'MAEtrain': ytr})
scipy.io.savemat('Results-MAEtest22-06.mat', {'MAEtest': yte})

