import os
import pandas as pd
from box import ConfigBox
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from CustomerChurn import logger
from CustomerChurn.utils.common import save_bin, create_directories, save_bin_dup
from CustomerChurn.entity.config_entity import ModelTrainerConfig
from scipy.stats import reciprocal, uniform

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


class ModelTrainer:
    """
    A class for training and selecting the best machine learning model based on the provided configuration.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer instance with the provided configuration and loads train and test data.

        Args:
        - config (ModelTrainerConfig): Configuration settings for model training.
        """
        self.config = config
        train = pd.read_csv(self.config.train_data_path)
        test = pd.read_csv(self.config.test_data_path)

        self.X_train= train.drop([self.config.target_col], axis=1)
        self.y_train = train[self.config.target_col]
        self.X_test = test.drop([self.config.target_col], axis=1)
        self.y_test = test[self.config.target_col]   


    def _randomized_search(self, name,clf,params, runs=50): 
        """
        Performs randomized search for hyperparameter tuning and evaluates the model.

        Args:
        - name (str): Name of the model for logging purposes.
        - clf: Machine learning model.
        - params: Hyperparameter grid or distribution for randomized search.
        - runs (int): Number of iterations for randomized search.

        Returns:
        - Tuple[Any, float]: Tuple containing the best model and its accuracy score.
        """
        rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=4, random_state=2, verbose=1)     

        rand_clf.fit(self.X_train, self.y_train) 
        best_model = rand_clf.best_estimator_
        
        best_score = rand_clf.best_score_
        logger.info("Trained with {} with score: {:.3f}".format(name, best_score))

        y_pred = best_model.predict(self.X_test)


        accuracy = accuracy_score(self.y_test, y_pred)
        logger.info('Predicted with {} ; Test score : {:.3f}'.format(name, accuracy))
        
        return best_model, accuracy


    def train(self):
        """
        Trains multiple machine learning models, selects the best one, and saves the model.

        Returns:
        - None
        """
        model_params = self.config.params

        models = ConfigBox({
            "Decision_Tree": {
                "model" : DecisionTreeClassifier(),
                "params" : model_params.Decision_Tree,
                "auto":{
                        "criterion" : ['gini', 'entropy'],
                        "splitter" : ['best', 'random'],
                        "max_depth" : range( 1, 32),
                        "min_samples_split" : uniform( 0.1, 1.0),
                        "min_samples_leaf" : uniform( 0.1, 0.5),
                        "max_features" : ['auto', 'sqrt', 'log2', None],
                }
            },
            "Random_Forest": {
                "model" : RandomForestClassifier(),
                "params" : model_params.Random_Forest,
                "auto":{
                    'n_estimators': range(10, 200),
                    'criterion': ['gini', 'entropy'],
                    'max_depth': range(1, 20),
                    'min_samples_split': range(2, 20),
                    'min_samples_leaf': range(1, 20),
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            "SVC": {
                "model" : SVC(),
                "params" : model_params.SVC,
                "auto":{
                    'C': reciprocal(0.1, 10),  
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': range(2, 7),
                    'gamma': ['scale', 'auto'] + list(uniform(0.1, 1.0).rvs(10)),
                    'coef0': uniform(-1, 1)
                }
            },
            "LogisticRegression":{
                "model" : LogisticRegression(),
                "params" : model_params.LogisticRegression,
                "auto" : {
                        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                        'C': uniform(0.1, 10), 
                        'fit_intercept': [True, False],
                        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                        'max_iter': range(50, 500, 50)
                }
            },
            "MultinomialNB":{
                "model" : MultinomialNB(),
                "params" : model_params.MultinomialNB,
                "auto": {
                        'alpha': uniform(0.1, 2.0), 
                        'fit_prior': [True, False],
                        'class_prior': [None, list(uniform(0.1, 1.0).rvs(3))] 
                    }
            },
            "GradientBoost":{
                "model": GradientBoostingClassifier(),
                "params" : model_params.GradientBoost,
                "auto": {
                        'learning_rate': uniform(0.01, 0.2),  # Uniform distribution for learning_rate
                        'n_estimators': range(50, 200, 30),
                        'max_depth': range(3, 10),
                        'min_samples_split': uniform(0.1, 1.0),  # Uniform distribution for min_samples_split
                        'min_samples_leaf': uniform(0.1, 0.5),   # Uniform distribution for min_samples_leaf
                        'subsample': uniform(0.5, 1.0),           # Uniform distribution for subsample
                        'max_features': ['auto', 'sqrt', 'log2', None],
                }
            },
            "AdaBoost":{
                "model" : AdaBoostClassifier(),
                "params" : model_params.AdaBoost,
                "auto":{
                        'n_estimators': range(50, 200),
                        'learning_rate': uniform(0.01, 1.0),  # Uniform distribution for learning_rate
                        'algorithm': ['SAMME', 'SAMME.R'],
                        'base_estimator': [None, DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
                }
            },
            "XGBoost":{
                "model" : XGBClassifier(),
                "params" : model_params.XGBoost,
                "auto":{
                        'learning_rate': uniform(0.01, 0.2),
                        'n_estimators': range(50, 200),
                        'max_depth': range(3, 10),
                        'min_child_weight': range(1, 10),
                        'subsample': uniform(0.5, 1.0),
                        'colsample_bytree': uniform(0.5, 1.0),
                        'gamma': uniform(0, 1),
                        'reg_alpha': uniform(0, 1),
                        'reg_lambda': uniform(0, 1),
                        'scale_pos_weight': range(1, 10),
                        'base_score': uniform(0.1, 0.9),
                        'booster': ['gbtree', 'gblinear', 'dart'],
                        'n_jobs': [-1],
                        'random_state': range(1, 100),
                }
            },
        })

        create_directories([os.path.join(self.config.root_dir, "models")])
        trained_models = []
        for model in models:
            clf = models[model].model
            params = models[model].params

            if self.config.auto_select:
                params = models[model].auto
            else:
                params = models[model].params
            if model=="XGBoost":
                clf_model, score = self._randomized_search(name=str(model) ,clf=clf, params=params, runs=300)
            else:
                clf_model, score = self._randomized_search(name=str(model) ,clf=clf, params=params)
            trained_models.append((clf_model, score))

            save_bin(data=clf_model, path=Path(os.path.join(self.config.root_dir, f"models/{str(model)}.joblib")))
        
        trained_models = sorted(trained_models, key=lambda x:x[1], reverse=True)  # [(model, score), (model, score), ..]
        best_model = trained_models[0][0]  # taking the model

        save_bin(data=best_model, path=Path(os.path.join(self.config.root_dir, self.config.model_name)))
        save_bin_dup(data=best_model, path=Path(os.path.join(self.config.permanent_path, "model.joblib")))

        best_model_name = str(best_model)[:str(best_model).find("(")]
        best_model_score = round(trained_models[0][1], 3)
        logger.info(f"Saved main model as {best_model_name}, with score - {best_model_score}")