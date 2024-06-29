# Author: @seekersoftec (GitHub)
# Created: 2024-02-08
# https://gist.github.com/seekersoftec/6d51a9801b24f094bc2212d46fcaa38e
import math
from pathlib import Path
import pickle
import joblib
import random
import numpy as np
import seaborn as sns
from typing import Optional, Tuple, Union, Dict, List, Any
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    auc,
    balanced_accuracy_score,
)
from sklearn.calibration import LabelEncoder
import matplotlib.pyplot as plt


def encode_labels(labels, encoder=None):
    """
    Encode a list of labels and return the label encoder object and the encoded labels.

    Parameters:
    labels (list): List of labels to be encoded.

    Returns:
    encoder (LabelEncoder): LabelEncoder object used for encoding.
    encoded_labels (list): Encoded labels.
    """
    encoder = LabelEncoder() if encoder is None else encoder
    # if encoder is not fitted, fit the encoder
    # if classes_ is not an attribute of the encoder, then fit the encoder
    if not hasattr(encoder, "classes_"):
        encoder.fit(labels)
    encoded_labels = encoder.transform(labels)

    return encoder, encoded_labels


def decode_labels(encoder, encoded_labels):
    """
    Decode encoded labels using the given label encoder object.

    Parameters:
    encoder (LabelEncoder): LabelEncoder object used for encoding.
    encoded_labels (list): Encoded labels to be decoded.

    Returns:
    decoded_labels (list): Decoded labels.
    """
    decoded_labels = encoder.inverse_transform(encoded_labels)
    label_mapping = {
        encoded_label: original_label
        for encoded_label, original_label in zip(encoded_labels, decoded_labels)
    }
    return decoded_labels, label_mapping


def roc_auc_score_multiclass(
    model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[int, Dict[Any, float], Dict[Any, float], Dict[Any, float], Dict[Any, float]]:
    """
    Calculate the ROC AUC score for a multiclass classification problem.

    Parameters:
    - model: The trained classifier model.
    - X_test: Testing data.
    - y_test: True class labels.

    Returns:
    - Tuple of containing number of classes(int) | fpr, tpr, and ROC AUC score for each class.
    """
    predictions = model.predict_proba(X_test)
    n_classes = predictions.shape[1]

    # Ensure y_test is an array (convert from list if needed)
    y_test = np.array(y_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(
            (y_test == i).astype(int), predictions[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])

    return n_classes, fpr, tpr, thresholds, roc_auc


class ModelSelection:
    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        scorer: Optional[Any] = None,
        multi_class: bool = False,
        shuffle_plot_colors: bool = False,
    ) -> None:
        """
        Initialize ModelSelection instance.

        For the plots, if the model(s) are not defined the best model is used.

        Parameters:
        - X_train, X_test: Training and testing data.
        - y_train, y_test: Training and testing labels.
        - scorer: Scorer to use for model evaluation in hyperparameter searches, defaults to None.
        - multi_class: if problem is a multi class problem, evaluate metrics for multi class, defaults to False.
        - shuffle_plot_colors: If True, shuffle the colors for plotting, defaults to False.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scorer = scorer
        self.multi_class = multi_class

        self.models: Dict[str, ClassifierMixin] = {}  # Dictionary to store models
        self.metrics: Dict[str, Dict[str, Any]] = (
            {}
        )  # Dictionary to store evaluation metrics
        self.best_model_name: str = ""
        # Colors for plotting
        self.COLORS: List[str] = [
            "brown",
            "coral",
            "green",
            "yellow",
            "black",
            "gray",
            "red",
            "blue",
            "purple",
            "pink",
            "orange",
            "olive",
            "skyblue",
            "cornflowerblue",
            "cyan",
            "aqua",
            "darkorange",
            "teal",
            "magenta",
            "tan",
        ]

        # Shuffle the colors
        if shuffle_plot_colors:
            random.shuffle(self.COLORS)

    def add_model(
        self, name: str, model: ClassifierMixin, replace: bool = False
    ) -> None:
        """
        Add a machine learning model to the selection.

        Parameters:
        - name: A unique name for the model.
        - model: The machine learning model to add.
        - replace: If True, replace the existing model with the same name.
        """
        if name in self.models and not replace:
            print(
                f"Warning: Model with name '{name}' already exists. Skipping addition."
            )
        else:
            self.models[name] = model

    def remove_model(self, name: str) -> None:
        """
        Remove a machine learning model from the selection.

        Parameters:
        - name: The name of the model to remove.
        """
        if name in self.models:
            del self.models[name]
        else:
            print(f"Warning: Model with name '{name}' does not exists.")

    def export_model(self, name: str, directory: Union[str, Path] = ".", _format="pkl"):
        """
        Export a machine learning model to a file.

        Parameters:
        - name (str): The name of the model to export.
        - directory (str | Path): The directory where the file will be saved. Default is the current directory.
        - _format (str): The format for saving the model. Supports "pkl" and "sav". Default is "pkl".
        """
        model = self.models[name]

        if _format == "pkl":
            with open(f"{directory}/{name}.pkl", "wb") as file:
                pickle.dump(model, file)
        elif _format == "sav":
            with open(f"{directory}/{name}.sav", "wb") as file:
                pickle.dump(model, file)
        else:
            raise ValueError(
                f"Unsupported format: {_format}. Supported formats are 'pkl' and 'sav'."
            )

    def import_model(self, name: str, directory: Union[str, Path] = ".", _format="pkl"):
        """
        Import a machine learning model from a file.

        Parameters:
        - name (str): The name of the model to import.
        - directory (str | Path): The directory where the file is located. Default is the current directory.
        - _format (str): The format used for saving the model. Supports "pkl" and "sav". Default is "pkl".

        Returns:
        - The imported machine learning model.
        """
        if _format == "pkl":
            with open(f"{directory}/{name}.pkl", "rb") as file:
                model = pickle.load(file)
        elif _format == "sav":
            with open(f"{directory}/{name}.sav", "rb") as file:
                model = pickle.load(file)
        else:
            raise ValueError(
                f"Unsupported format: {_format}. Supported formats are 'pkl' and 'sav'."
            )

        return model

    def train_models(self) -> None:
        """
        Train all added models and evaluate their performance.
        """
        total_models = len(self.models)

        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"Training model {i} of {total_models}: {name}")
            self.models[name] = model.fit(self.X_train, self.y_train)
            self.evaluate_model(name, model, self.X_test, self.y_test)
            print(f"Model {i} of {total_models} trained")

    def evaluate_model(
        self,
        name: str,
        model: ClassifierMixin,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a model using common classification metrics.

        Parameters:
        - name: Name of the model.
        - model: The trained machine learning model.
        - X_test: Testing data.
        - y_test: True labels for the test set.
        - save: If True, save metrics to the metrics dictionary.

        Returns:
        - Dictionary containing evaluation metrics.
        """
        predictions = model.predict(X_test)

        accuracy = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        cm = confusion_matrix(y_test, predictions)
        mislabeled_points = (y_test.flatten() != predictions.flatten()).sum()

        if self.multi_class:
            _, _, _, _, auc_score = roc_auc_score_multiclass(model, X_test, y_test)
        else:
            auc_score = roc_auc_score(
                np.array(y_test),
                np.array(predictions),
                average="weighted",
            )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc_score,
            "confusion_matrix": cm,
            "mislabeled_points": mislabeled_points,
        }

        if save:
            self.metrics[name] = metrics

        return metrics

    def get_best_model(
        self, metric: str = "f1_score"
    ) -> Tuple[ClassifierMixin, str, Dict[str, Any]]:
        """
        Get the best model based on a specified metric.

        Parameters:
        - metric: The metric to use for comparison. (accuracy | precision, recall, | f1_score | roc_auc)

        Returns:
        - Tuple containing the best model, its name, and corresponding metrics.
        """
        if (
            metric != "confusion_matrix"
            or metric != "mislabeled_points"
            or len(self.metrics) != 0
        ):
            self.best_model_name = max(
                self.metrics, key=lambda x: self.metrics[x][metric]
            )
            best_model = self.models[self.best_model_name]

            return best_model, self.best_model_name, self.metrics[self.best_model_name]

        return None

    def plot_confusion_matrix(
        self,
        model_name: Optional[str] = None,
        all_models=False,
        # all_models_title: str = "Confusion Matrix for All Models",
        file_path="plots/confusion_matrix.png",
    ) -> None:
        """
        Plot the confusion matrix for a given model in the models list.

        Parameters:
        - model_name: The name of the model to plot the confusion matrix for.
        """
        if not all_models:
            if model_name is None:
                _, model_name, _ = self.get_best_model()

            cm = self.metrics[model_name]["confusion_matrix"]

            # Plotting the confusion matrix with numbers
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                xticklabels=np.unique(self.y_test),
                yticklabels=np.unique(self.y_test),
            )

            plt.title(f"Confusion Matrix for {model_name}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()

        else:
            # Assuming 'models' is a dictionary containing your models
            num_models = len(self.models)
            grid_size = int(math.ceil(math.sqrt(num_models)))

            # Create a Subplots Grid
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), dpi=400)
            axes = axes.ravel()

            # Iterate Over Models
            for i, model_key in enumerate(self.models):
                metrics = self.metrics[model_key]
                cm = metrics["confusion_matrix"]

                # Plot Heatmap
                sns.heatmap(
                    cm,
                    annot=True,
                    ax=axes[i],
                    fmt="g",
                    cmap="Blues",
                    annot_kws={"size": 16},
                )

                # Set Title and Labels
                axes[i].set_title(
                    list(self.models.keys())[i], fontsize=16, fontweight="bold"
                )
                axes[i].set_xlabel("Predicted", fontsize=14, fontweight="bold")
                axes[i].set_ylabel("Actual", fontsize=14, fontweight="bold")

            # Remove Empty Plots
            for j in range(num_models, grid_size * grid_size):
                fig.delaxes(axes[j])

            # Adjust Layout and Save/Show
            fig.tight_layout()
            if len(file_path) > 0:
                plt.savefig(file_path)

            # plt.title(all_models_title)
            plt.show()

    def plot_bar_chart(
        self,
        model_name: Optional[str] = None,
        all_models: bool = False,
        palette: str = "viridis",
        # all_models_title: str = "Evaluation Metrics for All Models",
        file_path: str = "plots/accuracy_precision_recall_f1_score1.png",
    ) -> None:
        """
        Plot a bar chart comparing evaluation metrics for a given model or all models.

        Parameters:
        - model_name: The name of the model to plot the metrics for. If None, the best model is chosen.
        - all_models: If True, plot metrics for all models in the selection.
        - palette: The color palette to use for the bar chart.
        - file_path: Directory to save the resulting bar chart.
        """
        if not all_models:
            if model_name is None:
                _, model_name, _ = self.get_best_model()

            metrics = self.metrics[model_name]

            accuracy = metrics["accuracy"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1_score = metrics["f1_score"]

            plt.figure(figsize=(8, 6))
            # Plot Metrics as Bar Chart
            sns.barplot(
                x=["accuracy", "precision", "recall", "f1_score"],
                y=[accuracy, precision, recall, f1_score],
                palette=palette,
                alpha=0.8,
            )
            plt.title(f"Evaluation Metrics for {model_name}")
            plt.xlabel("Metrics")
            plt.ylabel("Score")
            plt.show()

        else:
            num_models = len(self.models)
            grid_size = int(math.ceil(math.sqrt(num_models)))

            # Create a Subplots Grid
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(25, 25), dpi=400)
            axes = axes.ravel()

            # Iterate Over Models
            for i, model_key in enumerate(self.models):
                metrics = self.metrics[model_key]
                accuracy = metrics["accuracy"]
                precision = metrics["precision"]
                recall = metrics["recall"]
                f1_score = metrics["f1_score"]

                # Plot Metrics as Bar Chart
                sns.barplot(
                    x=["accuracy", "precision", "recall", "f1_score"],
                    y=[accuracy, precision, recall, f1_score],
                    ax=axes[i],
                    palette=palette,
                    alpha=0.8,
                )

                # Increase labels font size
                axes[i].tick_params(axis="both", which="major", labelsize=14)
                axes[i].set_title(model_key, fontsize=16, fontweight="bold")
                axes[i].set_xlabel("Metric", fontsize=14, fontweight="bold")
                axes[i].set_ylabel("Score", fontsize=14, fontweight="bold")

            # Remove Empty Plots
            for j in range(num_models, grid_size * grid_size):
                fig.delaxes(axes[j])

            # Adjust Layout and Save/Show
            fig.tight_layout()
            if len(file_path) > 0:
                plt.savefig(file_path)

            # plt.title(all_models_title) # loc="right", y=-0.25, x=0.5
            plt.show()

    def plot_roc_curve(
        self,
        models: Optional[Dict[str, ClassifierMixin]] = None,
        multi_class: bool = False,
        file_path: Optional[str] = "plots/roc_curve.png",
        all_models: bool = False,
        # all_models_title: str = "ROC-AUC Curve for All Models"
    ) -> None:
        """
        Plot the ROC curve for each model in one figure.

        Parameters:
        - models: A dictionary of machine learning models with their names as keys.
        - multi_class: If True, evaluate AUC for multiclass. Default is False.
        - file_path: File path to save the resulting ROC curve plot. Default is "plots/roc_curve.png".
        - all_models: If True, plot ROC curves for all models. Default is False.
        """
        if models is None and not all_models:
            model, model_name, _ = self.get_best_model()

            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    f"Model {model_name} does not have a predict_proba method."
                )

            n_classes, fpr, tpr, thresholds, roc_auc = roc_auc_score_multiclass(
                model, self.X_test, self.y_test
            )

            # Plot ROC curve for each class
            plt.figure()
            for i, color in zip(range(n_classes), self.COLORS):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label=f"ROC curve (area = {roc_auc[i]:.2f}) for class {i}",
                )

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC-AUC Curve for {model_name}")
            plt.legend(loc="lower right", borderaxespad=0)  # Adjust the legend layout
            plt.show()

            return

        if all_models:
            models = self.models

        num_models = len(models)
        grid_size = int(math.ceil(math.sqrt(num_models)))

        # Create a Subplots Grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(25, 30), dpi=400)
        axes = axes.ravel()
        error_axes = []

        for i, (model_name, model) in enumerate(models.items()):
            try:
                if not hasattr(model, "predict_proba"):
                    raise AttributeError(
                        f"Model {model_name} does not have a predict_proba method."
                    )

                if multi_class:
                    n_classes, fpr, tpr, thresholds, roc_auc = roc_auc_score_multiclass(
                        model, self.X_test, self.y_test
                    )

                    # Plot ROC curve for each class
                    for j, color in zip(range(n_classes), self.COLORS):
                        axes[i].plot(
                            fpr[j],
                            tpr[j],
                            color=color,
                            lw=2,
                            label=f"ROC curve (area = {roc_auc[j]:.2f}) for class {j}",
                        )
                else:
                    predictions = model.predict_proba(self.X_test)
                    fpr, tpr, thresholds = roc_curve(np.array(self.y_test), predictions)
                    roc_auc = auc(fpr, tpr)

                    # plot the roc curve
                    axes[i].plot(
                        fpr,
                        tpr,
                        color=self.COLORS[i],
                        lw=2,
                        label="ROC curve (area = %0.2f)" % roc_auc,
                    )

                # plot the random line
                axes[i].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                # set the limits
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                # set the labels
                axes[i].set_xlabel(
                    "False Positive Rate", fontsize=14, fontweight="bold"
                )
                axes[i].set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
                axes[i].set_title(model_name, fontsize=16, fontweight="bold")
                axes[i].legend(
                    loc="lower right", borderaxespad=0
                )  # Adjust the legend layout

            except Exception as e:
                error_axes.append(i)
                print(f"Error for model {model_name}: {str(e)}")
                pass

        # Remove Empty Plots
        for j in range(num_models, grid_size * grid_size):
            fig.delaxes(axes[j])

        fig.tight_layout()

        if len(file_path) > 0:
            plt.savefig(file_path)

        # plt.title(all_models_title)
        plt.show()


# TODO: Add more metrics and visualizations as needed
# TODO: Add GridSearchCV to find the best hyperparameters for each model
# TODO: add support for DL models too

# def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):

#     # creating a set of all the unique classes using the actual class list
#     unique_class = set(actual_class)
#     roc_auc_dict = {}
#     for per_class in unique_class:
#         # creating a list of all the classes except the current class
#         other_class = [x for x in unique_class if x != per_class]

#         # marking the current class as 1 and all other classes as 0
#         new_actual_class = [0 if x in other_class else 1 for x in actual_class]
#         new_pred_class = [0 if x in other_class else 1 for x in pred_class]

#         # using the sklearn metrics method to calculate the roc_auc_score
#         roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
#         roc_auc_dict[per_class] = roc_auc

#     return roc_auc_dict

# from sklearn.metrics import roc_curve, auc


# # Ensure y_test is an array (convert from list if needed)
# self.y_test = np.array(self.y_test)

# # # Compute micro-average ROC curve and ROC area
# fpr, tpr, _ = roc_curve(self.y_test.ravel(), predictions.ravel())
# roc_auc = auc(fpr, tpr)

# def roc_auc_score_multiclass(actual_class, pred_class):
#     """
#     Calculate the ROC AUC score for a multiclass classification problem.

#     Parameters:
#     - actual_class: List of true class labels.
#     - pred_class: List of predicted class labels.

#     Returns:
#     - Dictionary containing the ROC AUC score for each class.
#     """
#     unique_class = set(actual_class)
#     roc_auc_dict = {}

#     for per_class in unique_class:
#         other_class = [x for x in unique_class if x != per_class]
#         new_actual_class = [0 if x in other_class else 1 for x in actual_class]
#         new_pred_class = [0 if x in other_class else 1 for x in pred_class]

#         # Compute ROC curve and ROC area
#         fpr, tpr, _ = roc_curve(new_actual_class, new_pred_class)
#         roc_auc = auc(fpr, tpr)

#         roc_auc_dict[per_class] = roc_auc

#     return roc_auc_dict

# import math

# # Assuming 'models' is a dictionary containing your models
# num_models = len(models)
# grid_size = int(math.ceil(math.sqrt(num_models)))

# # Create a Subplots Grid
# fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), dpi=400)
# axes = axes.ravel()

# # Iterate Over Models
# for i, model in enumerate(models):
#     metrics = modelSelector.metrics
#     cm = metrics[list(models.keys())[i]]['confusion_matrix']

#     # Plot Heatmap
#     sns.heatmap(cm, annot=True, ax=axes[i], fmt='g', cmap='Blues', annot_kws={"size": 16})

#     # Set Title and Labels
#     axes[i].set_title(list(models.keys())[i], fontsize=16, fontweight='bold')
#     axes[i].set_xlabel('Predicted', fontsize=14, fontweight='bold')
#     axes[i].set_ylabel('Actual', fontsize=14, fontweight='bold')

# # Remove Empty Plots
# for j in range(num_models, grid_size * grid_size):
#     fig.delaxes(axes[j])

# # Adjust Layout and Save/Show
# plt.tight_layout()
# plt.savefig("plots/confusion_matrix.png")
# plt.show()


# def plot_roc_curve(self, model_name: Optional[str] = None) -> None:
#     """
#     Plot the ROC curve for a given model.

#     Parameters:
#     - model_name: The model to plot the ROC curve for.
#     """
#     if model_name is None:
#         model, model_name, _ = self.get_best_model()

#     predictions = model.predict_proba(self.X_test)
#     n_classes = predictions.shape[1]

#     # Ensure y_test is an array (convert from list if needed)
#     self.y_test = np.array(self.y_test)

#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve((self.y_test == i).astype(int), predictions[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot ROC curve for each class
# plt.figure()
# for i, color in zip(range(n_classes), self.COLORS):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=2,
#         label=f"ROC curve (area = {roc_auc[i]:.2f}) for class {i}",
#     )

# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"Receiver Operating Characteristic (ROC) Curve for {model_name}")
# plt.legend(loc="lower right")
# plt.show()


# # This will create a figure with one row and two columns, making the subplots appear side by side.
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))


# # This will create subplots side by side for each model. Adjust the figsize parameter based on your preference for the width of the figure.
# fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), dpi=400)
# axes = axes.ravel()


# https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a
