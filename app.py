import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc,classification_report
from sklearn.impute import SimpleImputer
import openpyxl
import optuna
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Deployment", layout="wide")

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def auto_process_data(data):
    processed_data = data.copy()
    label_encoders = {}
    
    if processed_data.isnull().sum().sum() > 0:
        st.info("Automatically handling missing values...")
        
        num_cols = processed_data.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            processed_data[num_cols] = num_imputer.fit_transform(processed_data[num_cols])
        
        cat_cols = processed_data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                if processed_data[col].isnull().any():
                    most_frequent = processed_data[col].mode()[0]
                    processed_data[col].fillna(most_frequent, inplace=True)
    
    for column in processed_data.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        processed_data[column] = label_encoders[column].fit_transform(processed_data[column].astype(str))
    
    return processed_data, label_encoders

def get_model_configs():
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression())
            ]),
            'params': {
                'classifier__penalty':['l1','l2'],
                'classifier__C':[0.01,0.1,1],
                'classifier__max_iter': [100, 200],
                'classifier__solver':['liblinear','saga']
            }
        },
        'Support Vector Machine': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True))
            ]),
            'params': {
                'classifier__C': [0.001, 0.1, 1],
                'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
                'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'classifier__max_iter':[100,200]
            }
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier())
            ]),
            'params': {
                'classifier__n_estimators':[100,200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2,5,10],
                'classifier__min_samples_leaf':[1,2,4],
            }
        },
        'XgBoost':{
            'pipeline':Pipeline([
            ('scaled',StandardScaler()),
            ('classifier',XGBClassifier(use_label_encoder=False,eval_metric='logloss'))
            ]),
            'params':{
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__subsample': [0.8, 1.0]               
            }
        }
    }
    return models

def train_model(X_train, y_train, selected_model, progress_bar=None):
    models = get_model_configs()
    model_config = models[selected_model]
    
    with st.spinner(f"Training {selected_model}..."):
        grid_search = GridSearchCV(
            estimator=model_config['pipeline'],
            param_grid=model_config['params'],
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring="accuracy"
        )
        grid_search.fit(X_train, y_train)
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        return grid_search.best_estimator_, grid_search.best_score_
def objective(trial, X_train, y_train, model_name):
    models = get_model_configs()
    model_config = models[model_name]
    dataset_size = len(X_train)
    cv_folds = 5 if dataset_size > 1000 else (3 if dataset_size > 500 else min(2, dataset_size))
    params = {}

    if model_name == 'Logistic Regression':
        params = {
            'classifier__penalty': trial.suggest_categorical('classifier__penalty', ['l1', 'l2']),
            'classifier__C': trial.suggest_float('classifier__C', 0.01, 1.0, log=True),
            'classifier__solver': trial.suggest_categorical('classifier__solver', ['liblinear', 'saga']),
            'classifier__max_iter': trial.suggest_int('classifier__max_iter', 100, 200)
        }
    
    elif model_name == 'Support Vector Machine':
        params = {
            'classifier__C': trial.suggest_float('classifier__C', 0.001, 1.0, log=True),
            'classifier__kernel': trial.suggest_categorical('classifier__kernel', ['linear', 'rbf', 'sigmoid']),
            'classifier__gamma': trial.suggest_categorical('classifier__gamma', ['scale', 'auto', 0.01, 0.1, 1]),
            'classifier__max_iter': trial.suggest_int('classifier__max_iter', 100, 200)
        }
    
    elif model_name == 'Random Forest':
         params = {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 200),
            'classifier__max_depth': trial.suggest_categorical('classifier__max_depth', [None, 10, 20]),
            'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 10),
            'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 4)
        }
    elif model_name == 'XGBoost':
         params = {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 300),
            'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2, log=True),
            'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 10),
            'classifier__min_child_weight': trial.suggest_int('classifier__min_child_weight', 1, 6)
        }
    
    pipeline = model_config['pipeline'].set_params(**params)
    pipeline.fit(X_train, y_train)
    
    score = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy").mean()
    return score
def auto_train(X_train, y_train, X_test, y_test):
    models = get_model_configs()
    results = {}
    best_score = 0
    best_model = None
    best_model_name = None

    st.write("🔄 Training models with Optuna hyperparameter tuning...")

    progress_cols = st.columns(len(models))
    progress_bars = {model_name: progress_cols[i].progress(0.0) for i, model_name in enumerate(models)}

    for model_name in models.keys():
        st.write(f"🛠 Training {model_name}...")

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, model_name), n_trials=20)

        # Retrieve best parameters and train model
        best_params = study.best_params
        pipeline = models[model_name]['pipeline'].set_params(**best_params)
        pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {
            'model': pipeline,
            'cv_score': study.best_value,
            'test_accuracy': test_accuracy
        }

        progress_bars[model_name].progress(1.0)

        # Track best model
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = pipeline
            best_model_name = model_name

    # Display results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Cross-Validation Score': [results[model]['cv_score'] for model in results],
        'Test Accuracy': [results[model]['test_accuracy'] for model in results]
    }).sort_values('Test Accuracy', ascending=False)

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)

    st.success(f"🏆 Best model: **{best_model_name}** with accuracy: **{best_score:.2%}**")

    return best_model, best_model_name

def get_classification_report(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    return df
def evaluate_models(X_train, X_test, y_train, y_test):
    models = get_model_configs()
    
    results = {}

    plt.figure(figsize=(10, 6))
    
    num_classes = len(set(y_test))  # Determine if the target is binary or multiclass

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Handle predict_proba for multiclass classification
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if num_classes == 2:
                y_prob = y_prob[:, 1]  # Binary classification
            else:
                y_prob = None  # Ignore for multiclass
        else:
            y_prob = None  # If the model doesn't support predict_proba

        # Adjust 'average' based on the number of classes
        average_type = 'binary' if num_classes == 2 else 'weighted'
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average_type)
        recall = recall_score(y_test, y_pred, average=average_type)
        f1 = f1_score(y_test, y_pred, average=average_type)
        
        # ROC-AUC only for binary classification
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None and num_classes == 2 else None
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc
        }

        if y_prob is not None and num_classes == 2:  # ROC curve only for binary
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    if num_classes == 2:  # Plot ROC curve only for binary classification
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results).T
    results_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
    plt.show()
    
    return results_df

def main():
    st.title("🤖  Machine Learning Model Deployment")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Data Upload & Analysis", "Model Training","Visualisation", "Prediction"])
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None

    if page=="Home":
        st.title("🚀 AutoML: Effortless Machine Learning")
        st.markdown(
        """
        Welcome to **AutoML**, a powerful yet easy-to-use tool that automates the process of building and evaluating 
        machine learning models. Whether you're a beginner exploring data or an expert looking for quick model deployment, 
        AutoML simplifies the entire workflow.
        """
        )

        st.header("🔹 Features")
        st.markdown(
        """
        - **Automated Model Selection** – Let AutoML pick the best algorithm for your data.
        - **Hyperparameter Tuning** – Optimize model performance without manual tweaking.
        - **Data Preprocessing** – Handle missing values, scaling, encoding, and feature engineering.
        - **Performance Evaluation** – Compare models with key metrics and visualizations.
        - **Model Export** – Save trained models for deployment.
        """
        )

        st.header("🚀 Get Started")
        st.markdown(
        """
        1. **Upload your dataset** – Provide a CSV or Excel file with your data.
        2. **Select your target variable** – Choose the column to predict.
        3. **Let AutoML do the magic!** – Sit back and watch the automation work.
        """
        )

        st.header("📊 Visual Insights")
        st.markdown(
        """
        Explore interactive charts and performance metrics to make informed decisions. 
        Use visualizations to compare model accuracy, precision, recall, and other key statistics.
        """
        )

        st.success("Start automating your ML workflows now! 🎯")
        st.write('''Developed By Gourav Singh,Ankit Yadav,Pushpansh''')
  
    if page == "Data Upload & Analysis":
        st.header("📊 Data Upload & Analysis")
        
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            st.session_state.data = load_data(uploaded_file)
            
            if st.session_state.data is not None:
                st.session_state.processed_data, st.session_state.label_encoders = auto_process_data(st.session_state.data)
                
                st.success("Data loaded and automatically processed!")
                
                st.subheader("Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Number of rows: {st.session_state.data.shape[0]}")
                with col2:
                    st.info(f"Number of columns: {st.session_state.data.shape[1]}")
                with col3:
                    missing_values = st.session_state.data.isnull().sum().sum()
                    st.info(f"Missing values: {missing_values} (Automatically handled)")
                
                st.subheader("Original Data Preview")
                st.dataframe(st.session_state.data.head())
                
                st.subheader("Processed Data Preview")
                st.dataframe(st.session_state.processed_data.head())
                
                st.subheader("Statistical Description")
                st.dataframe(st.session_state.processed_data.describe())
                
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(st.session_state.processed_data.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
    
    elif page == "Model Training":
        st.header("🎯 Auto Model Training")
        
        if st.session_state.processed_data is None:
            st.warning("Please upload and process your data first!")
            return
            
        st.subheader("Select Features and Target")
        columns = st.session_state.processed_data.columns.tolist()
        
        st.session_state.features = st.multiselect("Select features", columns, default=columns[:-1])
        st.session_state.target = st.selectbox("Select target variable", columns)
        
        if st.button("Auto Train Models"):
            if len(st.session_state.features) > 0 and st.session_state.target:
                X = st.session_state.processed_data[st.session_state.features]
                y = st.session_state.processed_data[st.session_state.target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                st.session_state.model, st.session_state.model_name = auto_train(X_train, y_train, X_test, y_test)
                
                y_pred = st.session_state.model.predict(X_test)
                
                st.subheader("Best Model Performance")
                
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                st.text("Classification Report:")

                df_report = get_classification_report(y_test, y_pred)
                st.dataframe(df_report)
                
                if st.session_state.model_name == "Random Forest":
                    st.subheader("Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.features,
                        'Importance': st.session_state.model.named_steps['classifier'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                title='Feature Importance Plot')
                    st.plotly_chart(fig)
                
                model_data = {
                    'model': st.session_state.model,
                    'model_name': st.session_state.model_name,
                    'label_encoders': st.session_state.label_encoders,
                    'features': st.session_state.features,
                    'target': st.session_state.target
                }
                joblib.dump(model_data, 'model_data.joblib')
                st.download_button(
                    label="Download trained model",
                    data=open('model_data.joblib', 'rb'),
                    file_name='model_data.joblib',
                    mime='application/octet-stream'
                )
    elif page=="Visualisation":
        st.header("Model Visualisation")
        if st.session_state.model is None:
            st.warning("Please train a model first!")
            return
    
        if st.session_state.processed_data is not None and st.session_state.features and st.session_state.target:
            X = st.session_state.processed_data[st.session_state.features]
            y = st.session_state.processed_data[st.session_state.target]
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create visualization options
            viz_option = st.selectbox(
                "Select visualization type", 
                ["Model Comparison", "ROC Curves", "Confusion Matrix"]
            )
        
            if viz_option == "Model Comparison":
                st.subheader("Model Performance Metrics")
            
            # Train all models to compare
                models = get_model_configs()
                results = {}
            
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
                for i, (name, model_config) in enumerate(models.items()):
                    progress_text.text(f"Training {name}...")
                    pipeline = model_config['pipeline']
                    pipeline.fit(X_train, y_train)
                
                    y_pred = pipeline.predict(X_test)
                    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
                
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
                
                    results[name] = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-score": f1,
                        "ROC-AUC": roc_auc
                    }
                
                    progress_bar.progress((i + 1) / len(models))
            
                progress_text.empty()
            
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)

                fig = px.bar(
                    results_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score'), 
                    x='index', y='Score', color='Metric', 
                    barmode='group',
                    title='Model Comparison',
                    labels={'index': 'Model'}
                )
                st.plotly_chart(fig)
            
            elif viz_option == "ROC Curves":
                st.subheader("ROC Curves")
            
                models = get_model_configs()
            
                fig = plt.figure(figsize=(10, 6))
            
                for name, model_config in models.items():
                    pipeline = model_config['pipeline']
                    pipeline.fit(X_train, y_train)
                
                    if hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
                plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curves')
                plt.legend(loc="lower right")
            
                st.pyplot(fig)
            
            elif viz_option == "Confusion Matrix":
                st.subheader("Confusion Matrices")
            
                models = get_model_configs()
            
                if len(models) > 4:
                    st.warning("Showing confusion matrices for the first 4 models")
                    model_items = list(models.items())[:4]
                else:
                    model_items = list(models.items())
            
                num_models = len(model_items)
                cols = 2
                rows = (num_models + 1) // 2
            
                fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
                axes = axes.flatten() if num_models > 1 else [axes]
            
                for i, (name, model_config) in enumerate(model_items):
                    pipeline = model_config['pipeline']
                    pipeline.fit(X_train, y_train)
                
                    y_pred = pipeline.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
                    axes[i].set_title(f"{name} - Confusion Matrix")
                    axes[i].set_xlabel("Predicted")
                    axes[i].set_ylabel("Actual")
            
                for j in range(num_models, len(axes)):
                    fig.delaxes(axes[j])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.subheader("Current Model Performance")
            best_model_pred = st.session_state.model.predict(X_test)
        
            st.metric("Accuracy", f"{accuracy_score(y_test, best_model_pred):.2%}")
        
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precision", f"{precision_score(y_test, best_model_pred):.2%}")
                st.metric("F1 Score", f"{f1_score(y_test, best_model_pred):.2%}")
            with col2:
                st.metric("Recall", f"{recall_score(y_test, best_model_pred):.2%}")
                if hasattr(st.session_state.model, "predict_proba"):
                    best_proba = st.session_state.model.predict_proba(X_test)[:, 1]
                    st.metric("AUC", f"{roc_auc_score(y_test, best_proba):.2%}")
        
                else:
                    st.warning("Please load and preprocess your dataset before running evaluation.")


    elif page == "Prediction":
        st.header("🎲 Make Predictions")
        
        if st.session_state.model is None:
            st.warning("Please train a model first!")
            return
            
        st.subheader("Enter Feature Values")
        st.info(f"Using best model: {st.session_state.model_name}")
        
        input_data = {}
        for feature in st.session_state.features:
            if feature in st.session_state.label_encoders:
                options = st.session_state.label_encoders[feature].classes_
                value = st.selectbox(f"Select {feature}", options)
                input_data[feature] = st.session_state.label_encoders[feature].transform([value])[0]
            else:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            
            prediction = st.session_state.model.predict(input_df)
            
            if st.session_state.target in st.session_state.label_encoders:
                original_prediction = st.session_state.label_encoders[st.session_state.target].inverse_transform(prediction)
                st.success(f"Predicted {st.session_state.target}: {original_prediction[0]}")
            else:
                st.success(f"Predicted {st.session_state.target}: {prediction[0]}")
            
            proba = st.session_state.model.predict_proba(input_df)
            st.subheader("Prediction Probability")
            
            if st.session_state.target in st.session_state.label_encoders:
                classes = st.session_state.label_encoders[st.session_state.target].classes_
            else:
                classes = st.session_state.model.classes_
                
            proba_df = pd.DataFrame(
                proba,
                columns=classes
            )
            st.dataframe(proba_df)

if __name__ == "__main__":
    main()
