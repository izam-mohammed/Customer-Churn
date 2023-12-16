import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from CustomerChurn import logger
from CustomerChurn.utils.common import save_bin
from CustomerChurn.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        data = pd.read_csv(self.config.data_path)
        data.drop(["customerID"], axis=1, inplace=True)

        # convert total charges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        categorical_feautres = self.config.features.categorical
        numeric_features = self.config.features.numerical

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_feautres}")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler(feature_range=(0, 1)))
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
                ("normalizer", MinMaxScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_feautres),
            ]
        )

        preprocesed_data = preprocessor.fit_transform(data)
        preprocesed_df = pd.DataFrame(preprocesed_data, columns=data.columns)

        save_bin(data=preprocessor, path=Path(os.path.join(self.config.root_dir, self.config.encoder_name)))
        preprocesed_df.to_csv(os.path.join(self.config.root_dir, "encoded_data.csv"),index=False)
        logger.info(f"The data shape - {preprocesed_data.shape}")
        

    def split_data(self):
        data = pd.read_csv(os.path.join(self.config.root_dir, "encoded_data.csv"))
        
        X = data.drop(self.config.target_col, axis=1) 
        y = data[self.config.target_col] 

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2) 

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info("train data shape - {train.shape}")
        logger.info("test data shape - {test.shape}")