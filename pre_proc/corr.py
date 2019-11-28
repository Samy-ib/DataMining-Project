import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# data = pd.read_excel('CTG.xls', 'Data', skiprows=1)
def matrix(corr):
    # corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        xticklabels = True,
        yticklabels = True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    ) 
    plt.show()
