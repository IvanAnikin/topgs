
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input, concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

class DNN():

  def __init__(
      self,
      c_dim = (1), a_dim = 3, d_dim = 3, width = 120, height = 160, depth = 2, output_size = 3, critic=False,
      filters=(16, 32, 64), filters_dropouts=[0.2, 0.2, 0.2, 0.2], dropouts=[0.2, 0.2, 0.2], last_layer_activation="linear", layers_after_filters=(64, 32, 16)):
    super().__init__()

    self.c_dim = c_dim
    self.a_dim = a_dim
    self.d_dim = d_dim
    self.width = width
    self.height = height
    self.depth = depth
    self.output_size = output_size
    self.filters = filters
    self.filters_dropouts = filters_dropouts
    self.dropouts = dropouts
    self.last_layer_activation = last_layer_activation
    self.layers_after_filters= layers_after_filters

    self.model = self.create_model(critic)
    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    if not critic: self.model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

  def create_model(self, critic=False):

    models = [] # [mlp, cnn]
    if(self.numdata_not_empty()): models.append(self.create_mlp(regress=False))
    if(self.vid_not_empty()):     models.append(self.create_cnn(regress=False))

    if(len(models) > 1):
      combinedInput = concatenate([models[0].output, models[1].output])

      x = Dense(4, activation="relu")(combinedInput)
      actor = Dense(self.output_size, activation="linear")(x)

      if critic:
        critic = Dense(1)(x)
        model = Model(inputs=[models[0].input, models[1].input], outputs=[actor, critic])
      else: model = Model(inputs=[models[0].input, models[1].input], outputs=actor)

    elif(len(models) == 1):
      actor = Dense(self.output_size, activation=self.last_layer_activation)(models[0].output)
      model = Model(inputs=[models[0].input], outputs=actor)

    else: raise ValueError('Model input data must not be empty')

    return model

  def create_mlp(self, regress=False):    # contours, actions, distances

    combined_inputs_array = []
    inputs_array = []
    if(self.c_dim != 0):
      c_input = Input(shape=self.c_dim)
      c = Dense(8, activation="relu")(c_input)
      x = tf.keras.layers.Reshape((8,))(c)
      combined_inputs_array.append(x)
      inputs_array.append(c_input)
    if (self.a_dim != 0):
      a_input = Input(shape=self.a_dim)
      a = Dense(8, activation="relu")(a_input)
      combined_inputs_array.append(a)
      inputs_array.append(a_input)
    if (self.d_dim != 0):
      d_input = Input(shape=self.d_dim)
      d = Dense(8, activation="relu")(d_input)
      combined_inputs_array.append(d)
      inputs_array.append(d_input)

    combinedInput = concatenate(combined_inputs_array, axis=1)

    combined = Dense(4, activation="relu")(combinedInput)

    model = Model(inputs=inputs_array, outputs=combined)

    return model

  def create_cnn(self, regress=False):

    inputShape = (self.depth, self.width, self.height)
    chanDim = -1

    inputs = Input(shape=inputShape)

    for (i, f) in enumerate(self.filters):
      # if this is the first CONV layer then set the input
      # appropriately
      if i == 0:
        x = inputs
      # CONV => RELU => BN => POO
      x = Conv2D(f, (3, 3), padding="same")(x)
      x = Activation("relu")(x)
      if(self.filters_dropouts[i] != 0): x = Dropout(self.dropouts[i])(x)
      x = BatchNormalization(axis=chanDim)(x)
      if(i<len(self.filters)-1): x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dropout(self.dropouts[0])(x)

    for (i, f) in enumerate(self.layers_after_filters):
      x = Dense(f)(x)
      x = Activation("relu")(x)
      x = Dropout(self.dropouts[i+1])(x)
      x = BatchNormalization(axis=chanDim)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
      x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)
    return model

  def numdata_not_empty(self):
    return self.c_dim != 0 or self.a_dim != 0 or self.d_dim != 0

  def vid_not_empty(self):
    return self.height != 0 and self.width != 0 and self.depth != 0

