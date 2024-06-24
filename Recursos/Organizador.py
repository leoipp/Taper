import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


def output_df(file_name: str, df: pd.DataFrame) -> None:
    if file_name.endswith('.xlsx'):
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cubagem_Ajustado', index=False)
    elif file_name.endswith('.csv'):
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name + '.csv', index=False)


def nova_canvas(layout, fig, ax, tb=True):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.setParent(None)
            layout.removeWidget(widget)
    canvas = FigureCanvas(fig)
    if tb is True:
        toolbar = NavigationToolbar(canvas)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        ax.clear()


def plot_layout(layout, x, y, eixo_x, eixo_y):
    try:
        fig = plt.Figure()
        ax = fig.add_subplot()
        nova_canvas(layout, fig, ax, tb=True)
        ax.scatter(x, y, alpha=.5)
        ax.plot([2, 50], [2, 50], color='red', linewidth=.3)
        ax.set_xlabel(eixo_x)
        ax.set_ylabel(eixo_y)
        ax.set_title(f"Qualidade do ajuste")
        fig.tight_layout()
    except:
        pass

def plot_layout_hist(layout, x, eixo_x, eixo_y):
    try:
        fig = plt.Figure()
        ax = fig.add_subplot()
        nova_canvas(layout, fig, ax, tb=True)
        ax.hist(x)
        ax.set_xlabel(eixo_x)
        ax.set_ylabel(eixo_y)
        ax.set_title(f"Qualidade do ajuste")
        fig.tight_layout()
    except:
        pass