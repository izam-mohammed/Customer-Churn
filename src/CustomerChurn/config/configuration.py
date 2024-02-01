from CustomerChurn.constants import *
from CustomerChurn.utils.common import read_yaml, create_directories
from CustomerChurn.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig,
                                                DataTransformationConfig,
                                                ModelTrainerConfig,
                                                ModelEvaluationConfig,
                                                PredictionConfig,)

class ConfigurationManager:
    """
    A class for managing and providing configuration entities for different stages of the data science pipeline.

    Attributes:
    - config (dict): The loaded configuration from YAML files.
    - params (dict): The loaded parameters from YAML files.
    - schema (dict): The loaded schema from YAML files.

    Methods:
    - __init__: Initializes the ConfigurationManager with file paths for configuration, parameters, and schema.
    - get_data_ingestion_config: Returns a DataIngestionConfig object based on the loaded configuration.
    - get_data_validation_config: Returns a DataValidationConfig object based on the loaded configuration and schema.
    - get_data_transformation_config: Returns a DataTransformationConfig object based on the loaded configuration and schema.
    - get_model_trainer_config: Returns a ModelTrainerConfig object based on the loaded configuration and parameters.
    - get_model_evaluation_config: Returns a ModelEvaluationConfig object based on the loaded configuration and schema.
    - get_prediction_config: Returns a PredictionConfig object based on the loaded configuration and schema.

    Usage:
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    """

    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        """
        Initializes the ConfigurationManager with file paths for configuration, parameters, and schema.

        Args:
        - config_filepath (str): File path for the configuration YAML file.
        - params_filepath (str): File path for the parameters YAML file.
        - schema_filepath (str): File path for the schema YAML file.
        """

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns a DataIngestionConfig object based on the loaded configuration.

        Returns:
        - DataIngestionConfig: Configuration object for data ingestion.
        """
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
        """
        Returns a DataValidationConfig object based on the loaded configuration and schema.

        Returns:
        - DataValidationConfig: Configuration object for data validation.
        """
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
        """
        Returns a DataTransformationConfig object based on the loaded configuration and schema.

        Returns:
        - DataTransformationConfig: Configuration object for data transformation.
        """
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
        """
        Returns a ModelTrainerConfig object based on the loaded configuration and parameters.

        Returns:
        - ModelTrainerConfig: Configuration object for model training.
        """

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
            permanent_path=config.permanent_path,
            auto_select=config.auto_select,
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Returns a ModelEvaluationConfig object based on the loaded configuration and schema.

        Returns:
        - ModelEvaluationConfig: Configuration object for model evaluation.
        """
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
        """
        Returns a PredictionConfig object based on the loaded configuration and schema.

        Returns:
        - PredictionConfig: Configuration object for making predictions.
        """
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