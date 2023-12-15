from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.model_training import ModelTrainer
from CustomerChurn import logger


STAGE_NAME = "Model training stage"


class ModelTrainerTrainingPipeline:
    def __init__(self) -> None:
        """
        Initialize ModelTrainerTrainingPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the model trainer training pipeline.

        Returns:
            None
        """
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e