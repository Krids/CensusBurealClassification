"""
This file is responsible for the evaluation of the model on slices of the dataframe.

Name: Felipe Lana Machado
Date: 01/05/2022
"""

import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.project_paths import IMAGES_PATH

from model.model_training import ModelTraining

sns.set()

class ModelSliceEvaluation:

    def __init__(self) -> None:
        self.model_training = ModelTraining()

    def slice_metrics(self, column, X, y_true, y_pred):
        """
        Calculates column metrics
        """
        df = pd.concat([X[column].copy(), y_true], axis=1)
        df['salary_pred'] = y_pred

        metrics = []
        for categ in df[column].unique():
            precision, recall, fbeta = self.model_training.compute_model_metrics(
                df[df[column] == categ]['salary_pred'],
                df[df[column] == categ]['salary']
            )
            metrics.append([categ, precision, recall, fbeta])
            print(f"[INFO] {categ}: Precision = {precision:.3f}, Recall = {recall:.3f}, F-Beta = {fbeta:.3f}")

        return pd.DataFrame(
            metrics,
            columns=[
                'Category',
                'Precision',
                'Recall',
                'F1'])

    def evaluate_slices(self, file, model_pipe, column, X, y, split):
        """
        model on a slice of data
        """
        y_pred = self.model_training.inference(model_pipe, X)
        slice_df = self.slice_metrics(column, X, y, y_pred)

        self.plot_slice_metrics(
            slice_df,
            f"{column} column for {split} data",
            os.path.join(IMAGES_PATH, f"slice_metrics_{column}_{split}")
        )

        print(f"Model evaluation on {column} slice of train data", file=file)
        print(slice_df.to_string(index=False), file=file)
        print("", file=file)


    def plot_slice_metrics(self, df, title, save_path=None):
        """
        metrics in a bar plot
        """
        df = df.melt(id_vars=['Category'], value_vars=['Precision', 'Recall', 'F1'])

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x='variable', y='value', hue='Category', data=df)

        ax.set(title=title)
        ax.legend(loc='lower right')

        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

        for c in ax.containers:
            labels = [f'{v.get_height():.3f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
            if save_path:
                plt.savefig(f'{save_path}.png', bbox_inches='tight')