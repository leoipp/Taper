import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = pd.read_excel(r'G:\Downloads\aaa.xlsx')

# Filtrar as colunas de diâmetros e DAP
dap = data['DAP'].values
ht = data['HT'].values  # Supondo altura total de 30 metros para simplificação
heights = np.array([0, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

# Extraindo as colunas de diâmetro
diameter_columns = ['D0', 'D0,5', 'D1', 'D2', 'D4', 'D6', 'D8', 'D10', 'D12', 'D14', 'D16', 'D18', 'D20', 'D22', 'D24', 'D26', 'D28', 'D30']
diameters = data[diameter_columns].values

# Vetores para armazenar os dados empilhados
h_all = []
d_all = []
dap_all = []
ht_all = []

# Empilhando os dados
for i in range(len(dap)):
    for j in range(len(heights)):
        h_all.append(heights[j])
        d_all.append(diameters[i][j])
        dap_all.append(dap[i])
        ht_all.append(ht[i])

# Convertendo para arrays numpy
h_all = np.array(h_all)
d_all = np.array(d_all)
dap_all = np.array(dap_all)
ht_all = np.array(ht_all)

# Definindo a função do modelo de taper de Kozak
def kozak_taper(h, beta0, beta1, beta2, DAP, Ht):
    x = h / Ht
    return DAP * np.sqrt(beta0 + beta1 * x + beta2 * x**2)

# Ajuste do modelo usando curve_fit
popt, pcov = curve_fit(lambda h, beta0, beta1, beta2: kozak_taper(h, beta0, beta1, beta2, dap_all, ht_all), h_all, d_all, p0=[0.5, 0.5, 0.5])

# Parâmetros ajustados
beta0, beta1, beta2 = popt

print(f'Parâmetros ajustados: beta0 = {beta0}, beta1 = {beta1}, beta2 = {beta2}')
print(pcov)

# Visualização do ajuste
plt.figure(figsize=(10, 6))
for i in range(len(dap)):
    plt.plot(heights, diameters[i], 'o', label=f'Árvore {i+1} Dados reais')
    fitted_diameters = kozak_taper(heights, beta0, beta1, beta2, dap[i], ht[i])
    plt.plot(heights, fitted_diameters, '-', label=f'Árvore {i+1} Ajuste do modelo')

plt.xlabel('Altura (h)')
plt.ylabel('Diâmetro (d)')
plt.legend()
plt.show()
