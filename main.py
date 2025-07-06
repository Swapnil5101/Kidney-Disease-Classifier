from CNN_Classifier import logger
from CNN_Classifier.pipeline.data_ingestion_stage1 import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
except Exception as e:
    logger.exception(e)
    raise e