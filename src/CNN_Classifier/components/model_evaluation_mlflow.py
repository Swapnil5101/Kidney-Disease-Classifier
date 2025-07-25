import os
import tensorflow as tf
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from CNN_Classifier.entity.config_entity import EvaluationConfig
from CNN_Classifier.utils.common import read_yaml, create_directories, save_json
from pathlib import Path


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            class_mode="sparse",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        )
        
        return model
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        
    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }
        save_json(path=Path("scores.json"), data=scores)
        
    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        # mlflow.set_experiment("Kidney-Disease-Classifier_VGG16")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        dagshub.init(repo_owner='Swapnil5101', repo_name='Kidney-Disease-Classifier', mlflow=True)
        
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
             
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    model=self.model,
                    artifact_path="model",
                    registered_model_name="Kidney-Disease-Classifier_VGG16"
                )
            else:
                mlflow.keras.log_model(
                    model=self.model,
                    artifact_path="model"
                )
        