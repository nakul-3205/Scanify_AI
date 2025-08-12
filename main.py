from scanify_ai.components.data_ingestion import DataIngestion
from scanify_ai.exception.exception import CustomException
from scanify_ai.logging.log_config import logger
import sys
from scanify_ai.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataTransformationConfig
from scanify_ai.components.data_validation import DataValidation
from scanify_ai.components.data_transformation import DataTransformation
from scanify_ai.entity.config_entity import DataValidationConfig 

if __name__=='__main__':
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        dataIngestionconfig=DataIngestionConfig(trainingPipelineConfig)
        dataValidationconfig=DataValidationConfig(trainingPipelineConfig)
        dataIngestion=DataIngestion(dataIngestionconfig)
        logger.info('dataingestion initiated')
        dataIngestionartifact=dataIngestion.initiate_data_ingestion()
        logger.info('Data ingestion complete, entering data validation')
        data_validation=DataValidation(dataIngestionartifact,dataValidationconfig)
        logger.info('Data Validation Initiated')
        data_validation_artifact=data_validation.initiate_data_validation()
        logger.info('Data Validation completed ')
        # print(data_validation_artifact)
        # print(dataIngestionartifact)
        logger.info('Data transformation initiated')
        
        data_transformation_config=DataTransformationConfig(trainingPipelineConfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation,data_transformation_artifact)
        logger.info('Data transformation completed')
        
        

    except Exception as e:
        raise CustomException(e,sys)