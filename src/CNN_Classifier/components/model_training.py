import tensorflow as tf
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from CNN_Classifier.entity.config_entity import TrainingConfig
import time

tf.compat.v1.enable_eager_execution()  # Force Eager execution if eager not enabled


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        
        # Loaded with compile=False, and then recompiled below to AVOID GRAPH MODE and loading the
        # saved model in Eager Mode only, so as to avoid numpy related conflicts
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        

    def train_valid_generator(self):
        image_size = tuple(self.config.params_image_size[:-1])  # (height, width)
        batch_size = self.config.params_batch_size

        # Load datasets using the newer, eager-compatible API
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )

        # Normalization layer (rescale pixel values from [0, 255] to [0, 1])
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        self.train_generator = self.train_generator.map(lambda x, y: (normalization_layer(x), y))
        self.valid_generator = self.valid_generator.map(lambda x, y: (normalization_layer(x), y))

        # Optional augmentation if enabled
        if self.config.params_is_augmentation:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ])
            self.train_generator = self.train_generator.map(lambda x, y: (augmentation(x), y))

        # Performance optimization
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_generator = self.train_generator.prefetch(buffer_size=AUTOTUNE)
        self.valid_generator = self.valid_generator.prefetch(buffer_size=AUTOTUNE)


    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
    def train(self):
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )  