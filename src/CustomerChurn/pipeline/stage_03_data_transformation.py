from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.data_transformation import DataTransformation
from CustomerChurn import logger


STAGE_NAME = "Data Transformation stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        """
        Initialize DataTransformationTrainingPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the data tranformation training pipeline.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_tranformation_config = config.get_data_transformation_config()
        data_tranformation = DataTransformation(config=data_tranformation_config)
        data_tranformation.transform_data()
        data_tranformation.split_data()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
