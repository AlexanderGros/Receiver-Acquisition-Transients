# -*- coding: utf-8 -*-
# models.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Convolution2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

possible_mods = ['BPSK', 'QPSK', '8PSK', 'QAM16']
vec_len = 4096


def get_oshea_model(vec_len=vec_len, num_classes=len(possible_mods), dr=0.5, learning_rate=0.001):
    """
    Builds and returns the OShEA CNN model architecture.
    """
    channel_1 = Input(shape=(2, vec_len, 1), name="input1")
    x = Convolution2D(256, (1, 3), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
    x = Dropout(dr)(x)
    x = Convolution2D(80, (2, 3), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    out_1 = Flatten()(x)

    out = Dense(256, kernel_initializer="he_normal", activation="relu", name="dense0")(out_1)
    out = Dropout(dr)(out)
    out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

    model = Model(inputs=channel_1, outputs=out)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=["accuracy"])
    return model


def get_model_2(vec_len=vec_len, num_classes=len(possible_mods), dr=0.5, dr2=0.2, learning_rate=0.001):
    """
    Builds and returns Model 2 (IQ).
    """
    channel_1 = Input(shape=(2, vec_len, 1), name="input1")
    x = Convolution2D(256, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
    x = Dropout(dr)(x)
    x = Convolution2D(128, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    x = AveragePooling2D(pool_size=(1, 32), strides=(1, 32))(x)
    out_1 = Flatten()(x)

    out = Dense(128, kernel_initializer="he_normal", activation="relu", name="dense0")(out_1)
    out = Dropout(dr2)(out)
    out = Dense(64, kernel_initializer="he_normal", activation="relu", name="dense1")(out)
    out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

    model = Model(inputs=channel_1, outputs=out)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=["accuracy"])
    return model


def get_constellation_model(nb=128, num_classes=len(possible_mods), dr=0.5, learning_rate=0.001):
    """
    Builds and returns the constellation image model.
    """
    channel_1 = Input(shape=(nb, nb, 1), name="input1")
    x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
    x = Dropout(dr)(x)
    x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    out_1 = Flatten()(x)

    out = Dense(128, kernel_initializer="he_normal", activation="relu", name="dense1")(out_1)
    out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

    model = Model(inputs=channel_1, outputs=out)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=["accuracy"])
    return model
