import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Garay:
    def __init__(self, cubagem: pd.DataFrame) -> None:
        """
        Inicialização da classe de ajuste do modelo de Taper Garay
        :param cubagem_path: Caminho do arquivo de cubagem
        """
        self.cubagem = cubagem

    @staticmethod
    def garay_taper(I: np.array, b0: float, b1: float, b2: float, b3: float):
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
        epsilon = 1e-10  # Small constant to avoid division by zero or log of zero
        with np.errstate(over='ignore', invalid='ignore'):
            h_safe = np.where(h == 0, epsilon, h)  # Replace zeros in h with epsilon
            h_power_b3 = np.power(h_safe, b3)
            HT_power_neg_b3 = np.power(HT, -b3)
            term = 1 - b2 * h_power_b3 * HT_power_neg_b3
            valid_term = np.where(term > 0, term, epsilon)  # Ensure term is positive
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
        DAP = pd.to_numeric(self.cubagem[DAP_col], errors='coerce')
        HT = pd.to_numeric(self.cubagem[HT_col], errors='coerce')
        h = pd.to_numeric(self.cubagem[h_col], errors='coerce')
        d = pd.to_numeric(self.cubagem[d_col], errors='coerce')
        # Filtrar HT ou h zeros
        valid_indices = (HT.notnull()) & (h.notnull()) & (DAP.notnull()) & (d.notnull())
        DAP, HT, h, d = DAP[valid_indices], HT[valid_indices], h[valid_indices], d[valid_indices]
        return DAP, HT, h, d

    def vetorizar_aplicacao(self, ID: str, DAP_col: str, HT_col: str):
        """
        Vetoriza os dados de entrada.
        :param DAP_col: Nome da coluna DAP
        :param HT_col: Nome da coluna HT
        :return: arrays dos valores das colunas
        """
        DAP = pd.to_numeric(self.cubagem[DAP_col], errors='coerce')
        HT = pd.to_numeric(self.cubagem[HT_col], errors='coerce')
        id = self.cubagem[ID]
        # Filtrar HT ou h zeros
        valid_indices = (HT.notnull()) & (DAP.notnull())
        DAP, HT, id = DAP[valid_indices], HT[valid_indices], id[valid_indices]
        return DAP, HT, id

    def ajuste(self, dap: str, ht: str, h: str, d: str, initial_guess: list):
        DAP, HT, h, d = self.vetorizar(dap, ht, h, d)
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

    @staticmethod
    def garay_taper_h(I: np.array, b0: float, b1: float, b2: float, b3: float):
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

        valid_indices = (denominator != 0) & (numerator / denominator > 0)
        h = np.full_like(DAP, np.nan, dtype=np.float64)  # Initialize h with NaNs
        h[valid_indices] = (numerator[valid_indices] / denominator) ** (1 / b3)

        return h

    @staticmethod
    def decremento_array(array: np.array, decresed_by: float, val_min: float):
        """
        Decremento de array para os calculo de h chapéu
        :param array: input de arrays
        :param decresed_by: Valor de decremento, tamanho padrao de tora
        :param val_min: valor minimo de comprimento de tora
        :return: Nested array com valores ate o minimio de acordo com decremento
        """
        result = []
        for num in array:
            current = num
            temp_array = []
            while current > val_min:
                temp_array.append(float(current))
                current -= decresed_by
            if current < val_min:
                temp_array.append(float(val_min))
            else:
                temp_array.append(float(current))
            result.append(np.array(temp_array))
        return result

    def aplicacao_real(self, id, DAP: np.array, HT: np.array, d: float, comp_tora: float, h_min: float, params: list) -> np.array:
        h_chapeu = self.garay_taper_h((DAP, d, HT), *params)
        _h_chapeu_ = self.decremento_array(h_chapeu, comp_tora, h_min)

        id = np.concatenate([np.tile(id, len(h)) for id, h in zip(id, _h_chapeu_)])
        dap_ = np.concatenate([np.tile(dap, len(h)) for dap, h in zip(DAP, _h_chapeu_)])
        ht_ = np.concatenate([np.tile(ht, len(h)) for ht, h in zip(HT, _h_chapeu_)])
        _h_ = np.concatenate(_h_chapeu_)

        _d_chapeu_ = self.garay_taper((dap_, _h_, ht_), *params)

        as_ = [float((np.pi * dap ** 2) / 40000) for dap in _d_chapeu_]
        as_ = np.asarray(as_)

        _ = np.array([x if x is not None else np.nan for x in _d_chapeu_])
        indexes_d_min = np.where(np.isclose(_, d, atol=1e-9))[0]

        # calculando os comprimentos de toretes reais
        dif = -np.diff(_h_)
        dif = np.insert(dif, 0, 0)
        dif[indexes_d_min] = 0

        asmed = []
        for i in range(len(as_)):
            if i in indexes_d_min:
                avg = 0
            else:
                avg = (as_[i] + as_[i - 1]) / 2
            asmed.append(float(avg))

        na = np.where(np.isnan(_h_))[0]
        asmed = np.insert(asmed, na, np.nan)
        vol = asmed*dif
        return _d_chapeu_, _h_, ht_, dap_, dif, as_, asmed, vol, id




df = pd.read_excel(r'C:\Users\Leonardo\PycharmProjects\Taper\cubagem_compilado.xlsx')
g = Garay(df)
DAP, HT, h, d = g.vetorizar('DAP', 'HT', 'h', 'd')
params, pcov = g.ajuste('DAP', 'HT', 'h', 'd', [1.0, 0.1, 0.05, 1.5])
d_est = g.aplicacao_generalista(DAP, HT, h, params)

df = pd.read_excel(r'C:\Users\Leonardo\PycharmProjects\Taper\teste.xlsx')
g = Garay(df)
DAP, HT, id = g.vetorizar_aplicacao('Arv', 'DAP', 'HT')
print(params)
_d_chapeu_, _h_, ht_, dap_, dif, as_, asmed, vol, id = g.aplicacao_real(id, DAP, HT, 30, 3, 0.15, params)
df = pd.DataFrame({'arv': id, 'd_est': _d_chapeu_, 'h_est': _h_, 'ht': ht_, 'dap': dap_, 'dif': dif, 'as': as_, 'asmed': asmed, 'vol': vol})

grop = df.groupby('arv').sum().reset_index()
bins = pd.interval_range(start=0, end=5, freq=0.25, closed='right')
grop['classe'] = pd.cut(df['vol'], bins, include_lowest=True)

grop['classe'].value_counts().sort_index().plot(kind='bar')
plt.show()

# plt.scatter(d, d_est, label='Estimates vs Actual')
# plt.xlabel('Actual d')
# plt.ylabel('Estimated d')
# plt.title('Scatter plot of Actual vs Estimated d')
# plt.legend()
# plt.show()