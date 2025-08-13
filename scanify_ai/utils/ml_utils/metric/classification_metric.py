from scanify_ai.entity.artifact_entity import ClassificationMetricArtifact
from scanify_ai.exception.exception import CustomException
from sklearn.metrics import f1_score,precision_score,recall_score
import os,sys

def get_classification_metric (y_true,y_pred):
    try:
        model_f1score=f1_score(y_true,y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_recall_score=recall_score(y_true,y_pred)

        classification_metric=ClassificationMetricArtifact(f1_score=model_f1score,precision_score=model_precision_score,
                                                        recall_score=model_recall_score)

        return classification_metric

    except Exception as e:
        raise CustomException(e,sys)