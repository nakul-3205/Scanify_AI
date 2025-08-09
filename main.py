from scanify_ai.components.data_ingestion import DataIngestion
from scanify_ai.exception.exception import CustomException
from scanify_ai.logging.log_config import logger
import sys
from scanify_ai.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig


if __name__=='__main__':
    try:
           trainingPipelineConfig=TrainingPipelineConfig()
           dataIngestionconfig=DataIngestionConfig(trainingPipelineConfig)
           dataIngestion=DataIngestion(dataIngestionconfig)
           logger.info('dataingestion initiated')
           dataIngestionartifact=dataIngestion.initiate_data_ingestion()
           print(dataIngestionartifact)
           
    except Exception as e:
        raise CustomException(e,sys)