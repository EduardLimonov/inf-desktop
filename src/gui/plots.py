import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import f1_score, precision_score, recall_score


def plot_accuracy(p_pred, y_true):
    x = np.arange(0.05, 0.96, 0.01)

    df = pd.DataFrame({"Порог выживания": x})
    metrics = {
        "F1-метрика": f1_score,
        "Precision (чувствительность к умершим)": precision_score,
        "Recall (чувствительность к выжившим)": recall_score
    }
    metric_results = {m: [] for m in metrics}

    for edge in x:
        y_pred = (p_pred >= edge).astype(int)
        for metric_label in metrics:
            metric_results[metric_label].append(metrics[metric_label](y_true, y_pred))

    for m in metrics:
        df[m] = metric_results[m]

    fig = px.line(df, x="Порог выживания", y=list(metrics.keys()), title="Зависимость метрик от порога выживания",
                  color_discrete_sequence=["red", "green", "blue"])
    fig.update_traces(hovertemplate=None)
    fig.update_layout(yaxis_title="Метрика", legend_title="Метрики", hovermode="x")

    return fig


def plot_bar(p_pred, y_true, gt_label: str):
    fig = px.histogram(pd.DataFrame({"Оценка выживания": p_pred,
                                     gt_label: pd.Series(y_true).apply(lambda s: "выжил" if int(s) else "не выжил")}),
                       color_discrete_sequence=["#8fabf2", "#f28fa6"],
                       title="Распределение оценок выживания (тестовые данные)",
                       # pattern_shape=gt_label,
                       color=gt_label,
                       barmode="stack")
    fig.update_layout(bargap=0.2, yaxis_title="Количество пациентов", legend_title=gt_label, xaxis_title="Оценка",
                      hovermode="x")
    return fig


def plot_figures(p_pred, y_true, gt_label: str):
    return plot_accuracy(p_pred, y_true), plot_bar(p_pred, y_true, gt_label)

