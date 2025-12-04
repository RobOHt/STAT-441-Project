"""
Shared utilities for STAT 441 model training notebooks.

This module provides common functions for:
- Data loading (raw vectorized and engineered features)
- Preprocessing (PCA, scaling, train/test split)
- Model evaluation (confusion matrix, classification report, ROC curves)
- Results saving and loading for model comparison
"""

import numpy as np
import pandas as pd
import cv2
import json
import os
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Constants
CLASSES = ['left', 'forward', 'right']
CLASS_COLORS = {'left': '#e74c3c', 'forward': '#27ae60', 'right': '#3498db'}
RANDOM_STATE = 42

# Paths (relative to notebooks/models/)
DATA_DIR = Path('../../data')
RAW_DATA_DIR = DATA_DIR / 'raw'
RESULTS_DIR = DATA_DIR / 'model_results'


def load_raw_images():
    """
    Load raw grayscale images from the data/raw directory.
    
    Returns:
        X: np.array of shape (n_samples, 4096) - flattened 64x64 images
        y: np.array of shape (n_samples,) - class labels
    """
    X_list = []
    y_list = []
    
    for cls in CLASSES:
        class_dir = RAW_DATA_DIR / cls
        image_paths = sorted(glob(str(class_dir / '*.png')))
        
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X_list.append(img.flatten())
                y_list.append(cls)
    
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    print(f"Loaded {len(X)} raw images")
    print(f"  Shape: {X.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def load_engineered_features():
    """
    Load engineered features from data/engineered_features.csv.
    
    Returns:
        X: np.array of shape (n_samples, 38) - engineered features
        y: np.array of shape (n_samples,) - class labels
        feature_names: list of feature names
    """
    df = pd.read_csv(DATA_DIR / 'engineered_features.csv')
    
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Loaded {len(X)} samples with {len(feature_cols)} engineered features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, feature_cols


def load_data():
    """
    Load both raw vectorized and engineered features.
    
    Returns:
        dict with keys:
            - 'raw': (X_raw, y_raw)
            - 'engineered': (X_eng, y_eng, feature_names)
    """
    X_raw, y_raw = load_raw_images()
    X_eng, y_eng, feature_names = load_engineered_features()
    
    return {
        'raw': (X_raw, y_raw),
        'engineered': (X_eng, y_eng, feature_names)
    }


def apply_pca(X_train, X_test, variance_threshold=0.95):
    """
    Apply PCA to reduce dimensionality while retaining specified variance.
    
    Args:
        X_train: Training features
        X_test: Test features
        variance_threshold: Proportion of variance to retain (default 0.95)
    
    Returns:
        X_train_pca, X_test_pca, pca_model, n_components
    """
    # First fit PCA to determine number of components
    pca_full = PCA(n_components=min(X_train.shape[0], X_train.shape[1]))
    pca_full.fit(X_train)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Fit PCA with selected components
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA: {X_train.shape[1]} features -> {n_components} components")
    print(f"  Variance retained: {cumulative_variance[n_components-1]:.2%}")
    
    return X_train_pca, X_test_pca, pca, n_components


def preprocess_data(X, y, test_size=0.2, apply_pca_reduction=False, 
                    pca_variance=0.95, scale=True):
    """
    Preprocess data with train/test split, optional PCA, and scaling.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of data for testing (default 0.2)
        apply_pca_reduction: Whether to apply PCA (default False)
        pca_variance: Variance threshold for PCA (default 0.95)
        scale: Whether to standardize features (default True)
    
    Returns:
        dict with preprocessed data and metadata
    """
    # Encode labels
    le = LabelEncoder()
    le.fit(CLASSES)  # Ensure consistent encoding
    y_encoded = le.transform(y)
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, 
        stratify=y_encoded
    )
    
    print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")
    
    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Apply PCA if requested
    pca_model = None
    n_components = None
    if apply_pca_reduction:
        X_train, X_test, pca_model, n_components = apply_pca(
            X_train, X_test, pca_variance
        )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': le,
        'scaler': scaler,
        'pca': pca_model,
        'n_components': n_components,
        'classes': CLASSES
    }


def evaluate_model(model, X_test, y_test, model_name, feature_type, 
                   label_encoder=None, plot=True):
    """
    Comprehensive model evaluation with balanced metrics.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels (encoded)
        model_name: Name of the model for display
        feature_type: 'raw' or 'engineered'
        label_encoder: LabelEncoder for class names
        plot: Whether to show plots
    
    Returns:
        dict with all evaluation metrics
    """
    classes = label_encoder.classes_ if label_encoder else CLASSES
    n_classes = len(classes)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        # For SVM, use decision function
        decision = model.decision_function(X_test)
        if len(decision.shape) == 1:
            # Binary case
            y_proba = np.column_stack([1 - decision, decision])
        else:
            y_proba = decision
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    # Per-class metrics
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    
    # ROC-AUC (one-vs-rest)
    roc_auc = None
    if y_proba is not None:
        try:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', 
                                        multi_class='ovr')
        except Exception as e:
            print(f"  Warning: Could not compute ROC-AUC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Features: {feature_type}")
    print(f"{'='*60}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {f1_macro:.4f}")
    print(f"Weighted F1:       {f1_weighted:.4f}")
    print(f"Macro Precision:   {precision_macro:.4f}")
    print(f"Macro Recall:      {recall_macro:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC (macro):   {roc_auc:.4f}")
    
    print(f"\nPer-class metrics:")
    for i, cls in enumerate(classes):
        print(f"  {cls:10s}: F1={f1_per_class[i]:.4f}, "
              f"Precision={precision_per_class[i]:.4f}, "
              f"Recall={recall_per_class[i]:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=classes, yticklabels=classes)
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_title(f'Confusion Matrix\n{model_name} ({feature_type})', 
                          fontsize=13, fontweight='bold')
        
        # ROC Curves
        if y_proba is not None:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            for i, cls in enumerate(classes):
                if n_classes == 2 and i == 0:
                    continue  # Skip first class for binary
                
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                else:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                
                roc_auc_i = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'{cls} (AUC={roc_auc_i:.3f})',
                            color=CLASS_COLORS.get(cls, None))
            
            axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1].set_xlabel('False Positive Rate', fontsize=12)
            axes[1].set_ylabel('True Positive Rate', fontsize=12)
            axes[1].set_title(f'ROC Curves\n{model_name} ({feature_type})', 
                              fontsize=13, fontweight='bold')
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'ROC curves not available\n(no probabilities)', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('ROC Curves', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Return results dictionary
    results = {
        'model_name': model_name,
        'feature_type': feature_type,
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'f1_per_class': {cls: float(f1_per_class[i]) for i, cls in enumerate(classes)},
        'precision_per_class': {cls: float(precision_per_class[i]) for i, cls in enumerate(classes)},
        'recall_per_class': {cls: float(recall_per_class[i]) for i, cls in enumerate(classes)},
        'confusion_matrix': cm.tolist()
    }
    
    return results


def cross_validate_model(model, X, y, cv=5, scoring='f1_macro'):
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of folds (default 5)
        scoring: Scoring metric (default 'f1_macro')
    
    Returns:
        mean_score, std_score, all_scores
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    
    print(f"Cross-validation ({cv}-fold, {scoring}):")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return scores.mean(), scores.std(), scores


def save_results(results, model_name):
    """
    Save model results to JSON file for later comparison.
    
    Args:
        results: dict or list of dicts with evaluation results
        model_name: Name for the results file
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    filepath = RESULTS_DIR / f'{model_name}_results.json'
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(model_name):
    """
    Load model results from JSON file.
    
    Args:
        model_name: Name of the results file (without extension)
    
    Returns:
        dict with results
    """
    filepath = RESULTS_DIR / f'{model_name}_results.json'
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_results():
    """
    Load all saved model results.
    
    Returns:
        list of result dictionaries
    """
    all_results = []
    
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return all_results
    
    for filepath in RESULTS_DIR.glob('*_results.json'):
        with open(filepath, 'r') as f:
            result = json.load(f)
            if isinstance(result, list):
                all_results.extend(result)
            else:
                all_results.append(result)
    
    print(f"Loaded {len(all_results)} result entries")
    return all_results


def plot_model_comparison(results_list, metric='f1_macro', figsize=(14, 6)):
    """
    Create comparison bar chart for multiple models.
    
    Args:
        results_list: List of result dictionaries
        metric: Metric to compare (default 'f1_macro')
        figsize: Figure size
    """
    # Separate by feature type
    raw_results = [r for r in results_list if r['feature_type'] == 'raw']
    eng_results = [r for r in results_list if r['feature_type'] == 'engineered']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique model names
    model_names = list(dict.fromkeys([r['model_name'] for r in results_list]))
    x = np.arange(len(model_names))
    width = 0.35
    
    # Get metric values
    raw_values = []
    eng_values = []
    
    for name in model_names:
        raw_val = next((r[metric] for r in raw_results if r['model_name'] == name), 0)
        eng_val = next((r[metric] for r in eng_results if r['model_name'] == name), 0)
        raw_values.append(raw_val if raw_val else 0)
        eng_values.append(eng_val if eng_val else 0)
    
    bars1 = ax.bar(x - width/2, raw_values, width, label='Raw (PCA)', color='steelblue')
    bars2 = ax.bar(x + width/2, eng_values, width, label='Engineered', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', 
                       va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', 
                       va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def get_class_weights():
    """
    Get class weights for handling imbalanced data.
    
    Returns:
        dict mapping class index to weight
    """
    # Based on class distribution: left=1620, forward=7343, right=937
    # Total = 9900
    counts = {'left': 1620, 'forward': 7343, 'right': 937}
    total = sum(counts.values())
    n_classes = len(counts)
    
    # Compute balanced weights: n_samples / (n_classes * n_samples_per_class)
    weights = {}
    for i, cls in enumerate(CLASSES):
        weights[i] = total / (n_classes * counts[cls])
    
    return weights


def print_class_distribution(y, label_encoder=None):
    """
    Print class distribution of labels.
    
    Args:
        y: Labels (encoded or string)
        label_encoder: Optional LabelEncoder for decoding
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print("Class Distribution:")
    for val, count in zip(unique, counts):
        if label_encoder is not None:
            label = label_encoder.inverse_transform([val])[0]
        else:
            label = val
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

