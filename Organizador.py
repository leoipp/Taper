import pandas as pd


def output_df(file_name: str, df: pd.DataFrame) -> None:
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Cubagem_Ajustado', index=False)


class Melt:

    def __init__(self, cubagem: str, sheet_name: str) -> None:
        """
        Inicialização da classe de ajuste da base de dados para cálculos de Taper
        :param cubagem: Arquivo de cubagem (.xlsx)
        """
        self.cubagem = cubagem
        self.sheet_name = sheet_name

    def read_file(self) -> pd.DataFrame:
        """
        Leitura de arquivo de acordo com parametros de inicialização da função
        :return: pd.DataFrame inicializado
        """
        cubagem_df = pd.read_excel(self.cubagem, sheet_name=self.sheet_name)
        return cubagem_df

    def melt_down(self, cols_fixed: list, cols: list) -> pd.DataFrame:
        """
        Ajuste de dados para base de modelagem
        :param cols_fixed: Colunas de identificação de cubagem (ARV; DAP; HT)
        :param cols: Colunas de segmentação da tora (d0...dn)
        :return: Base de dados ajustada matricialmente
        """
        cubagem_df = self.read_file()
        melted_df = pd.melt(cubagem_df, id_vars=cols_fixed, value_vars=cols)
        # sorted_melted_df = melted_df.sort_values(by=[cols_fixed[0], 'variable'])
        return melted_df


cols = ['D0', 'D0,5', 'D1', 'D2', 'D4', 'D6', 'D8',
        'D10', 'D12', 'D14', 'D16', 'D18', 'D20',
        'D22', 'D24', 'D26', 'D28', 'D30', 'D32',
        'D34',	'D36',	'D38',	'D40',	'D42',	'D44',	'D46',	'D48']
cols_fixed = ['ARV', 'DAP', 'HT']
m = Melt('cubagem.xlsx', '>35').melt_down(cols_fixed, cols)
output_df('teste.xlsx', m)
