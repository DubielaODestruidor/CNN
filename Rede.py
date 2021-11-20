# Neste arquivo tem o método de buscar e testar uma imagem em específico.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.preprocessing.image import ImageDataGenerator

#Essas duas bibibliotecas são para normalizar a imagem nova de teste que chega
import numpy as np
from keras.preprocessing import image #biblioteca para normalizar a imagem nova que for testada.

#-------------------------------------------------------------------------

# Gerando os padrões da rede
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))# Não obrigatório
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

# A partir daqui estou teianando a rede.

base_treinamento = gerador_treinamento.flow_from_directory(r'archive\Tire Textures\training_data',
                                                           target_size = (64, 64),
                                                           batch_size = 1,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory(r'archive\Tire Textures\testing_data',
                                               target_size = (64, 64),
                                               batch_size = 1,
                                               class_mode = 'binary')
# steps_per_epoch = 4000 == POrque tenho 4000 imagens na base Treinamento.
classificador.fit_generator(base_treinamento, steps_per_epoch = 599 / 1,
                            epochs = 7, validation_data = base_teste,
                            
                            # validation_steps = 1000 porque tenho 1000 imagens nos testes.
                            validation_steps = 325 / 1)


imagem_teste = image.load_img(r'\archive\Tire Textures\testing_data\normal\IMG_4255.jpg', target_size = (64, 64))

imagem_teste = image.img_to_array(imagem_teste)
imagem_teste/=255
imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = classificador.predict(imagem_teste)

base_treinamento.class_indices

####
 




























