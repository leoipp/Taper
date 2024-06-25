from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Garay:
    def __init__(self, cubagem, dap, ht, h, d):
        if cubagem is not None and dap is not None and ht is not None and h is not None and d is not None:
            self.cubagem = cubagem
            self.dap, self.ht, self.h, self.d = self.vetorizar(dap, ht, h, d)
        else:
            self.dap = dap
            self.ht = ht
            self.h = h
            self.d = d
        self.initial_params = None
        self.optimized_params = None

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


    @staticmethod
    def clean_data(*arrays):
        mask = np.isfinite(arrays[0])
        for array in arrays[1:]:
            mask &= np.isfinite(array)
        return (array[mask] for array in arrays)

    @staticmethod
    def model_function(params, dap, h, ht): # GARAY_TAPER
        b0, b1, b2, b3 = params
        epsilon = 1e-10  # Small constant to avoid division by zero or log of zero
        dap = np.where(dap <= 0, epsilon, dap)  # Protect against non-positive values for log
        h_safe = np.where(h == 0, epsilon, h)  # Replace zeros in h with epsilon
        ht_safe = np.where(ht == 0, epsilon, ht)  # Replace zeros in HT with epsilon
        h_power_b3 = np.power(h_safe, b3)
        ht_power_neg_b3 = np.power(ht_safe, -b3)
        term = 1 - b2 * h_power_b3 * ht_power_neg_b3
        valid_term = np.where(term > epsilon, term, epsilon)  # Ensure term is positive
        d_est = dap * b0 * (1 + b1 * np.log(valid_term))
        return np.maximum(d_est, epsilon)  # Ensure diameters are non-negative

    @staticmethod
    def garay_taper_h(params, dap, d, ht):
        """
        Equação ajustada para h utilizando o metodo de Taper de Garay
        :param params: Lista de parâmetros ajustados [b0, b1, b2, b3]
        :param dap: DAP (diâmetro à altura do peito)
        :param d: diâmetro em certo ponto
        :param ht: altura total
        :return: float ou array de alturas em certo d min
        """
        b0, b1, b2, b3 = params
        epsilon = 1e-10  # Small constant to avoid division by zero or log of zero
        dap = np.where(dap <= 0, epsilon, dap)  # Protect against non-positive values for log
        ht_safe = np.where(ht == 0, epsilon, ht)  # Replace zeros in HT with epsilon

        C = np.exp(-1 / b1)
        numerator = ht_safe ** b3 * (1 - C * np.exp(d / (dap * b0 * b1)))
        denominator = b2

        valid_term = np.where(denominator != 0, numerator / denominator, epsilon)
        valid_term = np.where(valid_term > epsilon, valid_term, epsilon)  # Ensure term is positive
        h_est = np.power(valid_term, 1 / b3)

        return np.maximum(h_est, epsilon)  # Ensure heights are non-negative

    @staticmethod
    def objective_function(params, dap, h, ht, d):
        try:
            predicted_d = Garay.model_function(params, dap, h, ht)
            mse = mean_squared_error(d, predicted_d)
        except ValueError as e:
            mse = np.inf
        return mse

    def find_initial_parameters(self):
        self.dap, self.ht, self.h, self.d = self.clean_data(self.dap, self.ht, self.h, self.d)
        initial_guess = derive_initial_guesses(self.dap, self.ht, self.h, self.d)

        result = minimize(self.objective_function, initial_guess, args=(self.dap, self.h, self.ht, self.d),
                          bounds=[(0.1, 10), (0.01, 1), (0.01, 1), (0.1, 2)], method='L-BFGS-B',
                          options={'maxfun': 10000, 'maxiter': 10000})
        if result.success:
            self.initial_params = result.x
        else:
            raise RuntimeError("Optimization did not converge: " + result.message)
        return self.initial_params

    def fit_model(self): # AJUSTE
        if self.initial_params is None:
            raise RuntimeError("Initial parameters not found. Please run find_initial_parameters() first.")

        bounds = [(0.1, 20), (0.001, 5), (0.001, 5), (0.01, 10)]

        result = differential_evolution(self.objective_function, bounds, args=(self.dap, self.h, self.ht, self.d),
                                        maxiter=30000, popsize=15, mutation=(0.5, 1), recombination=0.7)

        if not result.success:
            result = minimize(self.objective_function, self.initial_params, args=(self.dap, self.h, self.ht, self.d),
                              bounds=bounds, method='Powell', options={'maxfev': 30000, 'maxiter': 30000})
        if not result.success:
            result = minimize(self.objective_function, self.initial_params, args=(self.dap, self.h, self.ht, self.d),
                              bounds=bounds, method='TNC', options={'maxfev': 30000, 'maxiter': 30000})
        if result.success:
            self.optimized_params = result.x
        else:
            raise RuntimeError("Model fitting did not converge: " + result.message)
        return self.optimized_params

    def predict(self, params=None):
        if self.optimized_params is None:
            raise RuntimeError("Model not fitted. Please run fit_model() first.")
        if params is None:
            return self.model_function(self.optimized_params, self.dap, self.h, self.ht)
        if params is not None:
            return self.model_function(*params, self.dap, self.h, self.ht)

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


def derive_initial_guesses(dap, ht, h, d):
    # Use linear regression to estimate initial b0
    X = np.vstack([dap, h, ht]).T
    y = d
    reg = LinearRegression().fit(X, y)
    b0 = reg.coef_[0]

    # Set other parameters to reasonable defaults
    b1 = 0.1
    b2 = 0.05
    b3 = 1.5

    return [b0, b1, b2, b3]


# df = pd.read_excel(r'G:\Downloads\Dados_Inservivel\Cubagem_geral_compilado.xlsx')
# taper = Garay(df, 'DAPmed', 'Ht', 'seccao', 'value')
# initial_params = taper.find_initial_parameters()
# print(initial_params)
# optimized_params = taper.fit_model()
# print(optimized_params)
# predict = taper.predict()
# print(predict)
#
# plt.scatter(taper.d, predict, label='Predicted vs Actual')
# plt.xlabel('Actual Diameter')
# plt.ylabel('Predicted Diameter')
# plt.title('Actual vs Predicted Diameters using Garay Taper Model')
# plt.legend()
# plt.show()