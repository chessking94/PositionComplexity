import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Sequential 

lr = 0.001

class MLP:
    def __init__(self, learning_rate, classification): # Input - learning_rate, boolean classification [T] or regression [F]
        self.learning_rate = learning_rate
        if classification:
            self.model = self.build_classification_model()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        else:
            self.model = self.build_regression_model()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                    loss='mse', metrics=['mae', 'mse'])

    def build_classification_model(self):
        model = Sequential()
        
        model.add(Dense(64, input_dim=832))
        model.add(Activation('relu'))

        model.add(Dense(15))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('sigmoid'))

        """        
        model.add(Dense(1048, input_dim=832))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(500))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(2))
        model.add(Activation('softmax'))
        """

        #model.summary()
        return model

    def build_regression_model(self):
        model = Sequential()
        
        
        model.add(Dense(64, input_dim=832))
        model.add(Activation('relu'))

        #model.add(Dense(15))
        #model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('relu'))
        
        """
        model.add(Dense(1048, input_dim=832))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(500))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(1))
        """        

        #model.summary()
        return model

    def train(self, x, y, val_x, val_y):
        history = self.model.fit(x, y, epochs=1, verbose=1, validation_data=(val_x, val_y))
        return history