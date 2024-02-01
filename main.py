from CustomerChurn import logger
from CustomerChurn.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CustomerChurn.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from CustomerChurn.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from CustomerChurn.pipeline.stage_04_model_training import ModelTrainerTrainingPipeline
from CustomerChurn.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


def run_pipeline(stage_name, pipeline_instance):
    """
    Run a specific stage of the sentiment analysis pipeline.

    Parameters:
    - stage_name: str
        Name of the pipeline stage.
    - pipeline_instance: object
        Instance of the pipeline stage to be executed.

    Returns:
        None
    """
    try:
        logger.info(f">>>>>> Stage {stage_name} started <<<<<<")
        pipeline_instance.main()
        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    run_pipeline("Data Ingestion", DataIngestionTrainingPipeline())
    run_pipeline("Data Validation", DataValidationTrainingPipeline())
    run_pipeline("Data Transformation", DataTransformationTrainingPipeline())
    run_pipeline("Model Training", ModelTrainerTrainingPipeline())
    run_pipeline("Model Evaluation", ModelEvaluationTrainingPipeline())
