import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
roc_curve, precision_recall_curve, auc)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

logging.basicConfig(level=logging.INFO)


def data_compile(qa_path, eval_path):
    qa_df = pd.read_csv(qa_path)
    eval_df = pd.read_csv(eval_path)

    merged_df = qa_df.merge(eval_df[['Question', 'multiple_choice_accuracy']],
                                on='Question',
                                how='inner')

    merged_df['multiple_choice_accuracy'] = merged_df['multiple_choice_accuracy'].astype(int)

    return merged_df


def extract_features(df):
    features = {}

    features['question_length'] = df['Question'].str.len()
    features['question_word_count'] = df['Question'].str.split().str.len()

    for col in ['A', 'B', 'C', 'D']:
        features[f'{col}_length'] = df[col].str.len()
        features[f'{col}_word_count'] = df[col].str.split().str.len()

    answer_lengths = df[['A', 'B', 'C', 'D']].apply(lambda x: x.str.len())
    features['answer_length_range'] = answer_lengths.max(axis=1) - answer_lengths.min(axis=1)
    features['mean_answer_length'] = answer_lengths.mean(axis=1)
    features['std_answer_length'] = answer_lengths.std(axis=1)

    features['has_technical_terms'] = df['Question'].str.contains(
        r'(model|python|math|py)',
        case=False
    ).astype(int)

    features['correct_answer_length'] = df.apply(
        lambda x: len(x[x['Correct Answer']]), axis=1
    )

    features['num_documents'] = df['Documents'].str.count(',') + 1

    return pd.DataFrame(features)

def get_models():
    # Individual models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ('Random Forest', rf),
            ('Gradient Boosting', gb),
            ('Logistic Regression', lr),
            ('SVM', svm),
            ('Decision Tree', dt),
            ('MLP', mlp)
        ],
        voting='soft'
    )
    stacking_clf = StackingClassifier(
        estimators=[
            ('Random Forest', rf),
            ('Gradient Boosting', gb),
            ('SVM', svm),
            ('MLP', mlp)
        ],
        final_estimator=LogisticRegression()
    )

    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Logistic Regression': lr,
        'SVM': svm,
        'Decision Tree': dt,
        'MLP': mlp,
        'Ensemble': ensemble,  # Add the ensemble model,
        'Stacking Classifier': stacking_clf
    }

    return models


# Additional Helper Functions
def plot_confusion_matrix(y_test, y_pred, class_names, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)  # Save the plot
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, output_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig(output_path)  # Save the plot
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, output_path='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(output_path)  # Save the plot
    plt.show()

def evaluate_models(X, y, models):
    results = {}
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    for name, model in models.items():
        logging.info(f"Evaluating {name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        cv_results = cross_validate(
            pipeline, X, y,
            cv=10,
            scoring=scoring,
            return_train_score=True
        )

        results[name] = {
            'test_accuracy': cv_results['test_accuracy'].mean(),
            'test_accuracy_std': cv_results['test_accuracy'].std(),
            'test_precision': cv_results['test_precision'].mean(),
            'test_recall': cv_results['test_recall'].mean(),
            'test_f1': cv_results['test_f1'].mean()
        }

    return results


def train_final_model(X, y, best_model_name, models):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', models[best_model_name])
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None

    feature_importance = None
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)

    plot_confusion_matrix(y_test, y_pred, class_names=np.unique(y), output_path='results_no_context/confusion_matrix.png')

    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, output_path='results/roc_curve.png')
        plot_precision_recall_curve(y_test, y_pred_proba, output_path='results_no_context/precision_recall_curve.png')

    logging.info("\nFinal Model Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))

    if feature_importance is not None:
        logging.info("\nFeature Importance:")
        logging.info(feature_importance.head(10).to_string())
        feature_importance.to_csv('feature_importance.csv', index=False)

    return pipeline, X_test, y_test, y_pred, feature_importance

def result_plots(cv_results):
    results_df = pd.DataFrame(cv_results).T
    results_df = results_df.reset_index().rename(columns={'index': 'Model'})

    results_df.to_csv('results_no_context/cross_validation_results_no_context.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Model', y='test_accuracy', ci=None)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.savefig('results_no_context/model_accuracy_comparison.png')
    plt.show()

    metrics = ['test_precision', 'test_recall', 'test_f1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y=metric, ci=None)
        plt.title(f'Model {metric.capitalize()} Comparison')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.savefig('results_no_context/model_{metric}_comparison.png')  # Save the plot
        plt.show()

def main():
    merged_df = data_compile('../question_generation/questions_and_answers.csv', 'evaluation_metrics.csv')
    X = extract_features(merged_df)
    y = merged_df['multiple_choice_accuracy']

    models = get_models()
    cv_results = evaluate_models(X, y, models)
    result_plots(cv_results)
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['test_accuracy'])
    logging.info(f"Best model: {best_model_name}")
    train_final_model(X, y, best_model_name, models)


if __name__ == "__main__":
    main()