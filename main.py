from CNN_Classifier import logger
from CNN_Classifier.pipeline.data_ingestion_stage1 import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.base_model_stage2 import BaseModelTrainingPipeline
from CNN_Classifier.pipeline.model_training_stage3 import ModelTrainingPipeline
from CNN_Classifier.pipeline.model_evaluation_mlflow_stage4 import EvaluationPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f"**********************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = BaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training Stage"
try:
    logger.info(f"**********************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f"**********************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
except Exception as e:
    logger.exception(e)
    raise e