import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def clean_text(text):
    # lowercase
    text = text.lower()
    # no urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def plotly_confusion_matrix(y_true, y_pred, labels=["fake", "real"], title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    z = cm
    x = labels
    y = labels

    z_text = [[str(cell) for cell in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=z_text,
        texttemplate="%{text}",
        colorscale="Blues",
        hoverinfo="text"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(constrain="domain"),
        yaxis=dict(autorange="reversed"),
        width=500,
        height=500
    )

    fig.show()