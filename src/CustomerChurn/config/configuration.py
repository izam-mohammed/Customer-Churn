from CustomerChurn.constants import *
from CustomerChurn.utils.common import read_yaml, create_directories
from CustomerChurn.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig,
                                                DataTransformationConfig,
                                                ModelTrainerConfig,
                                                ModelEvaluationConfig,
                                                PredictionConfig,)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        target_col = self.schema.TARGET_COLUMN
        nan_ratio = self.schema.COL_NAN_RATIO
        categories = self.schema.CATEGORIES

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
            target_col=target_col.name,
            nan_ratio=nan_ratio,
            categories=categories,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        all_cols = list(self.schema.COLUMNS.keys())
        target_col = self.schema.TARGET_COLUMN
        features = self.schema.FEATURES

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            encoder_name=config.encoder_name,
            test_size=config.test_size,
            all_cols=all_cols,
            target_col=target_col.name,
            features=features,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        target_col = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            params = self.params,
            target_col=target_col.name,
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            target_column = schema.name,
            models_dir = config.models_dir,
           
        )

        return model_evaluation_config
    
    
    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        target_column = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        prediction_config = PredictionConfig(
            root_dir = config.root_dir,
            model_path= config.model_path,
            vectorizer_path=config.vectorizer_path,
            data_path=config.data_path,
            prediction_file=config.prediction_file,
            target_column=target_column.name,
           
        )

        return prediction_config