import os
from pathlib import Path
from zipfile import ZipFile
from urllib import request
import tensorflow as tf
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def download_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        
        self.save_model(path=self.config.base_model_path, model=self.model)
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        if not path.parent.exists():
            os.makedirs(path.parent)
        model.save(path)
        
    @staticmethod   
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            # Freeze layers till the specified index 
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
                
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        prediction = tf.keras.layers.Dense(
            units=classes, 
            activation='softmax'
        )(flatten_in)
        
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        
        full_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        
        full_model.summary()    
        return full_model
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )