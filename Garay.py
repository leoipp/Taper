import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Garay:
    def __init__(self, cubagem_path: str) -> None:
        """
        Inicialização da classe de ajuste do modelo de Taper Garay
        :param cubagem_path: Caminho do arquivo de cubagem
        """
        self.cubagem = pd.read_excel(cubagem_path)

    def garay_taper(self, I: np.array, b0: float, b1: float, b2: float, b3: float):
        """
        Equação ajustada para d utilizando o metodo de Taper de Garay
        :param I: Variáveis independentes (DAP, h, HT)
        :param b0: Parâmetro beta 0
        :param b1: Parâmetro beta 1
        :param b2: Parâmetro beta 2
        :param b3: Parâmetro beta 3
        :return: float ou array do diâmetro estimado em h
        """
        DAP, h, HT = I
        with np.errstate(over='ignore', invalid='ignore'):
            h_power_b3 = np.power(h, b3)
            HT_power_neg_b3 = np.power(HT, -b3)
            term = 1 - b2 * h_power_b3 * HT_power_neg_b3
            valid_term = np.where(term > 0, term, np.nan)  # Tendo certeza que o termo é positivo
            result = DAP * b0 * (1 + b1 * np.log(valid_term))
        return result

    def vetorizar(self, DAP_col: str, HT_col: str, h_col: str, d_col: str):
        """
        Vetoriza os dados de entrada.
        :param DAP_col: Nome da coluna DAP
        :param HT_col: Nome da coluna HT
        :param h_col: Nome da coluna h
        :param d_col: Nome da coluna d
        :return: arrays dos valores das colunas
        """
        DAP = self.cubagem[DAP_col].values
        HT = self.cubagem[HT_col].values
        h = self.cubagem[h_col].values
        d = self.cubagem[d_col].values
        # Filtrar HT ou h zeros
        valid_indices = (HT != 0) & (h != 0)
        DAP, HT, h, d = DAP[valid_indices], HT[valid_indices], h[valid_indices], d[valid_indices]
        return DAP, HT, h, d

    def plot_data(self, x: str, y: str, line: bool):
        """
        Plota os dados de cubagem para análise visual.
        """
        if line:
            plt.scatter(self.cubagem[x], self.cubagem[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Figura 01')
            plt.show()
        else:
            plt.scatter(self.cubagem[x], self.cubagem[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Figura 01')
            plt.show()

    def ajuste(self, initial_guess: list):
        DAP, HT, h, d = self.vetorizar('DAP', 'HT', 'h', 'd')
        popt, pcov = curve_fit(self.garay_taper, (DAP, h, HT), d, p0=initial_guess)
        return popt, pcov

    def aplicacao_generalista(self, DAP: np.array, HT: np.array, h: np.array, params: list) -> np.array:
        """
        Aplica o modelo ajustado aos dados de entrada.
        :param DAP: Array de DAP
        :param HT: Array de HT
        :param h: Array de alturas
        :param params: Lista de parâmetros ajustados [b0, b1, b2, b3]
        :return: Array de diâmetros estimados
        """
        return self.garay_taper((DAP, h, HT), *params)

    def garay_taper_h(self, I: np.array, b0: float, b1: float, b2: float, b3: float):
        """
        Equação ajustada para h utilizando o metodo de Taper de Garay
        :param I: Variáveis independentes (DAP, h, HT)
        :param b0: Parâmetro beta 0
        :param b1: Parâmetro beta 1
        :param b2: Parâmetro beta 2
        :param b3: Parâmetro beta 3
        :return: float ou array do h em certo d min
        """
        DAP, d, HT = I
        C = np.exp(-1 / b1)
        numerator = HT ** b3 * (1 - C * np.exp(d / (DAP * b0 * b1)))
        denominator = b2
        h = (numerator / denominator) ** (1 / b3)
        return h

    def aplicacao_real(self, DAP: np.array, HT: np.array, d: np.array, comp_tora: float, params: list) -> np.array:
        h_chapeu = self.garay_taper_h((DAP, d, HT), *params)
        d_chapeu = self.garay_taper((DAP, h_chapeu, HT), *params)
        qtd_toras = h_chapeu/comp_tora



# Exemplos de uso
# g = Garay('teste.xlsx')
# g.plot_data('d', 'h', False)
#
# DAP, HT, h, d = g.vetorizar('DAP', 'HT', 'h', 'd')
# params, pcov = g.ajuste([1.0, 0.1, 0.05, 1.5])
# aplicacao = g.aplicacao_generalista(DAP, HT, h, params)
#
#
# plt.scatter(d, aplicacao)
# plt.plot([0, 50], [0, 50], color = 'red', linewidth = .5)
# plt.xlabel('d')
# plt.ylabel('dest')
# plt.title('Figura 01')
# plt.show()
