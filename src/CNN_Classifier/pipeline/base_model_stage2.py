from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.base_model import PrepareBaseModel

STAGE_NAME = "Prepare Base Model Stage"

class BaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.download_base_model()
        prepare_base_model.update_base_model()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n------------")
        
    except Exception as e:
        logger.exception(e)
        raise e