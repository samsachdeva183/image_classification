from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

def models_factory(model_type, image_size):

    if model_type == "vgg16":
        base_model = applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "vgg19":
        base_model = applications.VGG19(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "resnet50":
        base_model = applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "inceptionv3":
        base_model = applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "xception":
        base_model = applications.Xception(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "mobilenet":
        base_model = applications.MobileNet(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "inceptionresnetv2":
        base_model = applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))
    elif model_type == "nasnet":
        base_model = applications.nasnet.NASNetLarge(weights = 'imagenet', include_top = False, input_shape = (image_size[0], image_size[1], 3))

    for layer in base_model.layers:
        layer.trainable = False
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1024, kernel_initializer = 'glorot_uniform', activation='relu'))
    top_model.add(Dense(1024, kernel_initializer = 'glorot_uniform', activation='relu'))
    top_model.add(Dense(1, activation = 'sigmoid'))
    model = Model(input = base_model.input, output = top_model(base_model.output))

    return model, base_model