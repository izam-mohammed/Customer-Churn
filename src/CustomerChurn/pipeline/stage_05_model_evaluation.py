from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.model_evaluation import ModelEvaluation
from CustomerChurn import logger


STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self) -> None:
        """
        Initialize ModelEvaluationTrainingPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the model evaluation training pipeline.

        Returns:
        None
        """
        config = ConfigurationManager()
        model_trainer_config = config.get_model_evaluation_config()
        model_trainer_config = ModelEvaluation(config=model_trainer_config)
        model_trainer_config.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
