import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import multiprocessing
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class HyperSearch:
    def __init__(self, X, y, log_name, max_concurrent_processes=5, scaler_type='StandardScaler'):
        """
        Initialize the HyperSearch class for hyperparameter optimization.

        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Target variable matrix (assumed preprocessed/scaled).
            log_name (str): Path to the log file to store results.
            max_concurrent_processes (int): Maximum number of concurrent processes.
            scaler_type (str): Choice of scaler - 'StandardScaler' or 'MinMaxScaler'.
        """
        self.X = X
        self.y = y
        self.log_name = log_name
        self.max_concurrent_processes = max_concurrent_processes

        # Choose scaler based on user input
        if scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'StandardScaler' or 'MinMaxScaler'")

        # Hyperparameter search space
        self.seed_arr = [44, 55, 66]  # Random seeds
        self.lrate_arr = [1e-3, 1e-4]  # Learning rates

        # Architectures: Shallow & Wide
        self.num_layers_arr = [1, 2, 3, 4]  # Number of layers for wide
        self.units_range = range(100, 501, 100)  # Units per layer for wide

        # Architectures: Narrow & Deep
        self.num_layers_arr_nd = [5, 10, 15, 20]  # Narrow and deep layers
        self.units_arr_nd = [5, 10, 15, 20, 25]  # Units per layer for deep networks

        self.units_per_layer_arr = self.generate_architectures()

        # Processed data placeholders
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    def preprocess_data(self):
        """
        Preprocess data: split into train, validation, and test sets, and scale features.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def generate_architectures(self):
        """
        Generate both shallow-wide and narrow-deep architectures.
        """
        architectures = []

        # Shallow & Wide architectures
        for num_layers in self.num_layers_arr:
            architectures.extend(product(self.units_range, repeat=num_layers))

        # Narrow & Deep architectures
        for num_layers_nd in self.num_layers_arr_nd:
            for units_nd in self.units_arr_nd:
                architectures.append((units_nd,) * num_layers_nd)

        return architectures

    def build_model(self, units_per_layer, lrate):
        """
        Build and train a TensorFlow neural network.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.X_train.shape[1:]))

        for units in units_per_layer:
            model.add(tf.keras.layers.Dense(units=units, activation="relu"))

        model.add(tf.keras.layers.Dense(self.y_train.shape[1]))  # Output layer
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lrate)

        stop_early = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
        model.compile(loss="mse", optimizer=optimizer)

        history = model.fit(
            self.X_train, self.y_train, 
            epochs=5000, validation_data=(self.X_val, self.y_val),
            callbacks=[stop_early], batch_size=int(np.sqrt(len(self.X_train))),
            verbose=0
        )
        return model, len(history.history['loss'])

    def cust_loss(self, y_pred, y_test):
        """
        Custom loss function: 99.5th - 0.5th percentile of prediction ratios.
        """
        ratio = (y_pred.flatten() / (y_test.flatten() + 1e-15)) - 1
        return np.percentile(ratio, 99.5) - np.percentile(ratio, 0.5)

    def train_and_evaluate(self, config):
        """
        Train a model for a specific hyperparameter configuration.
        """
        units_per_layer, lrate, seed = config
        tf.keras.utils.set_random_seed(seed)

        print(f"Training: Units={units_per_layer}, Learning Rate={lrate}, Seed={seed}")
        model, n_epochs = self.build_model(units_per_layer, lrate)

        y_pred = model.predict(self.X_test)
        test_mse = model.evaluate(self.X_test, self.y_test, verbose=0)
        ratio_loss = self.cust_loss(y_pred, self.y_test)

        # Log the results
        with open(self.log_name, "a") as f:
            f.write(f"{list(units_per_layer)}, {lrate}, {seed}, {n_epochs}, {test_mse}, {ratio_loss}\n")

        tf.keras.backend.clear_session()

    def run_search(self):
        """
        Run the hyperparameter search in parallel using multiprocessing.
        """
        self.preprocess_data()
        tasks = [
            (units, lrate, seed)
            for units in self.units_per_layer_arr
            for lrate in self.lrate_arr
            for seed in self.seed_arr
        ]

        active_processes = []
        for task in tasks:
            process = multiprocessing.Process(target=self.train_and_evaluate, args=(task,))
            process.start()
            active_processes.append(process)

            # Limit concurrent processes
            while len(active_processes) >= self.max_concurrent_processes:
                for p in active_processes:
                    if not p.is_alive():
                        active_processes.remove(p)
                time.sleep(1)

        for p in active_processes:
            p.join()
