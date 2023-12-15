from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.data_validation import DataValidation
from CustomerChurn import logger


STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        """
        Initialize DataIngestionTrainingPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the data ingestion training pipeline.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.final_validation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e