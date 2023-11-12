import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy import stats

def plot_roc(models_prob: dict, y_true, save_to=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (model_name, y_pred_prob) in enumerate(models_prob.items()):
        RocCurveDisplay.from_predictions(
            y_true,
            y_pred_prob,
            name=model_name,
            plot_chance_level=False,
            ax=ax
        )
        if model_name == 'random':
            ax.lines[i].set_linestyle('--')

    plt.axis('square')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves of expected goals")
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, 'roc.png'))    
    
    plt.show()
    return fig

def plot_goal_rate(models_prob: dict, y_true, save_to=None):
    percentile_list = np.arange(0, 101, 5)/100
    for (model_name, y_pred_prob) in models_prob.items():
        df = pd.DataFrame({'y_pred_prob': y_pred_prob, 'is_goal': y_true})
        centiles = pd.qcut(df['y_pred_prob'], q=percentile_list, duplicates='drop')
        goal_rate = df.groupby(centiles).is_goal.mean()

        plt.plot(percentile_list[:-1]*100, goal_rate*100, '--' if model_name == 'random' else '-', label=model_name)

    plt.xlabel("Shot probability model percentile")
    plt.ylabel("Goal Rate")
    plt.ylim((0, 100))
    plt.yticks(np.arange(0, 101, 10))
    plt.gca().set_yticklabels([f'{y}%' for y in plt.gca().get_yticks()]) 

    plt.xlim((100, 0))
    plt.xticks(np.arange(0, 101, 10))
    plt.title("Goal Rate for each percentile")
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, 'goal_rate.png'))    
    
    plt.show()
    return fig
    
def plot_cumulative_goal(models_prob: dict, y_true, save_to=None):
    for i, (model_name, y_pred_prob) in enumerate(models_prob.items()):
        centiles = 100 * stats.rankdata(y_pred_prob, 'min') / len(y_pred_prob)
        df = pd.DataFrame({'y_pred_centiles': centiles, 'y_true': y_true})
        df = df.loc[df.y_true == 1]
        ax = sns.ecdfplot(data=df, x=100-df.y_pred_centiles, label=model_name, stat='percent')

        if model_name == 'random':
            ax.lines[i].set_linestyle('--')

    plt.ylabel("Proportion of cumulated goals")
    plt.ylim((0, 100))
    plt.yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f'{y}%' for y in ax.get_yticks()])  

    plt.xlabel("Shot probability model percentile")
    plt.xlim((-5, 105))
    xTicks = np.arange(0, 101, 10)
    _, xTicksLabels =  plt.xticks(xTicks)
    plt.xticks(np.arange(0, 101, 10), [100 - int(item.get_text()) for item in xTicksLabels])

    plt.title("Cumulative % of goals")
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, 'cumulative_goals.png'))    
    
    plt.show()
    return fig

def plot_calibration_curve(models_prob: dict, y_true, save_to=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (model_name, y_pred_prob) in enumerate(models_prob.items()):
        CalibrationDisplay.from_predictions(
            y_true,
            y_pred_prob,
            name=model_name,
            ax=ax,
            n_bins=40,
            ref_line=i==0
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curves for each model")
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, 'calibration.png'))
    
    plt.show()
    return fig

def plot_all(models_prob: dict, y_true, save_to_folder=None):
    if save_to_folder is not None and not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)

    # Generate random uniform distribution for reference
    y_random_prob = np.random.uniform(0.0, 1.0, len(y_true))
    models_prob['random'] = y_random_prob

    plot_roc(models_prob, y_true, save_to_folder)
    plot_calibration_curve(models_prob, y_true, save_to_folder)
    plot_goal_rate(models_prob, y_true, save_to_folder)
    plot_cumulative_goal(models_prob, y_true, save_to_folder)
