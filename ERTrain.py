from KerasModels.inception_resnet_v2  import InceptionResNetV2
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import os,shutil
import warnings
from keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from keras.optimizers import Adam
import time
import json
execution_path = os.getcwd()
initial_learning_rate = 0.01 
def lr_schedule(epoch):
    """
    Learning Rate Schedule
    """
 
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

def ERtrainModel(Data_Path):
    global DATASET_DIR,DATASET_TRAIN_DIR,DATASET_TEST_DIR,DATASET_Json_DIR,DATASET_Logs_DIR,DATASET_MODELs_DIR
    DATASET_DIR = os.path.join(execution_path, Data_Path)
    DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")
    DATASET_Json_DIR = os.path.join(DATASET_DIR, "json")
    DATASET_Logs_DIR = os.path.join(DATASET_DIR, "logs")
    DATASET_MODELs_DIR = os.path.join(DATASET_DIR, "models")
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)
    
    if not os.path.isdir(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            
    if  os.path.isdir(DATASET_MODELs_DIR):
        shutil.rmtree(DATASET_MODELs_DIR)

    if not os.path.isdir(DATASET_TRAIN_DIR):
        os.makedirs(DATASET_TRAIN_DIR)
        
  
    if not os.path.isdir(DATASET_TRAIN_DIR):
        os.makedirs(DATASET_TRAIN_DIR)
        
    if not os.path.isdir(DATASET_TEST_DIR):
        os.makedirs(DATASET_TEST_DIR)

    if not os.path.isdir(DATASET_MODELs_DIR):
        os.makedirs(DATASET_MODELs_DIR)
        
    if not os.path.isdir(DATASET_Json_DIR):
        os.makedirs(DATASET_Json_DIR)
        
    if not os.path.isdir(DATASET_Logs_DIR):
        os.makedirs(DATASET_Logs_DIR)
    if not os.path.isdir(DATASET_TestModel_DIR):
        os.makedirs(DATASET_TestModel_DIR)
   
    if not os.path.isdir(DATASET_TestModel_timeDIR):
        os.makedirs(DATASET_TestModel_timeDIR)
    
 

def trainModel(modelType, num_objects, num_experiments, enhance_data=False, batch_size = 32, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224, continue_from_model=None, transfer_from_model=None, transfer_with_full_training=None, initial_num_objects = None, save_full_model = False):
      
        """
                 'trainModel()' function starts the model actual training. It accepts the following values:
                 - num_objects , which is the number of classes present in the dataset that is to be used for training
                 - num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
                 - enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
                 - batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted. The value is set to 32 by default, but can be increased or decreased depending on the meormory of the compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
                 - initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised to keep this value as it is if you don't have deep understanding of this concept.
                 - show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it. It is set to False by default
                 - training_image_size(optional) , this value is used to define the image size on which the model will be trained. The value is 224 by default and is kept at a minimum of 100.
                - continue_from_model (optional) , this is used to set the path to a model file trained on the same dataset. It is primarily for continuos training from a previously saved model.
                - transfer_from_model (optional) , this is used to set the path to a model file trained on another dataset. It is primarily used to perform tramsfer learning.
                - transfer_with_full_training (optional) , this is used to set the pre-trained model to be re-trained across all the layers or only at the top layers.
                - initial_num_objects (required if 'transfer_from_model' is set ), this is used to set the number of objects the model used for transfer learning is trained on. If 'transfer_from_model' is set, this must be set as well.
                - save_full_model ( optional ), this is used to save the trained models with their network types. Any model saved by this specification can be loaded without specifying the network type.

                 *

                :param num_objects:
                :param num_experiments:
                :param enhance_data:
                :param batch_size:
                :param initial_learning_rate:
                :param show_network_summary:
                :param training_image_size:
                :param continue_from_model:
                :param transfer_from_model:
                :param initial_num_objects:
                :param save_full_model:
                :return:
                """
        
        
        num_classes = num_objects
        num_epochs = num_experiments
        if modelType==InceptionResNetV2:
           m_type="InceptionResNetV2"
        if modelType==InceptionResNetV2:
            m_type="InceptionResNetV2"
                 
        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100

       
        if modelType==InceptionResNetV2:   
            if (continue_from_model != None):
                continue_from_model= os.path.join(DATASET_DIR, "InceptionResNetV2-ERmodel.h5")
                model = InceptionResNetV2(input_shape=(training_image_size, training_image_size,3), weights=continue_from_model, classes=num_classes,
                include_top=True)
                if (show_network_summary == True):
                    print("Training using weights from a previouly model")
            elif (transfer_from_model != None):
                transfer_from_model= os.path.join(DATASET_DIR, "InceptionResNetV2-ERmodel.h5")
                base_model = InceptionResNetV2(input_shape=(training_image_size, training_image_size, 3), weights= transfer_from_model,
                include_top=False, pooling="avg")

                network = base_model.output
                network = Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = keras.models.Model(inputs=base_model.input, outputs=network)
                

                if (show_network_summary == True):
                    print("Training using weights from a pre-trained ImageNet model")
            else:
                base_model = InceptionResNetV2(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_classes,
                include_top=False, pooling="avg")

                network = base_model.output
                network = Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = keras.models.Model(inputs=base_model.input, outputs=network)
              
        
        
        
       
        optimizer =  Adam(lr=initial_learning_rate, decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
      
       
       
        
        if (show_network_summary == True):
            model.summary()

        save_weights_condition = True

        if(save_full_model == True ):
            save_weights_condition = False
        elif(save_full_model == False):
            save_weights_condition = True
        if m_type=="InceptionResNetV2":     
                  
            if(continue_from_model != None):
                model_name = 'InceptionResNetV2-continue-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
            
            elif(transfer_from_model != None):
                model_name = 'InceptionResNetV2-imagenet-transfer-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
            
            else:
                model_name = 'InceptionResNetV2-ERmodel_ex-{epoch:03d}_acc-{accuracy:03f}.h5'
 
 
        model_path = os.path.join(DATASET_MODELs_DIR, model_name)

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     monitor='accuracy',
                                     verbose=1,
                                     save_weights_only=save_weights_condition,
                                     save_best_only=True,
                                     period=1)

        
        log_name = 'lr-{}_{}'.format(initial_learning_rate,time.strftime("%Y-%m-%d-%H-%M-%S"))
        logs_path = os.path.join(DATASET_Logs_DIR, log_name)

        tensorboard = TensorBoard(log_dir=logs_path, 
                            histogram_freq=0, 
                            write_graph=False, 
                            write_images=False)
        
        if (enhance_data == True):
            print("Using Enhanced Data Generation")

        height_shift = 0
        width_shift = 0
        if (enhance_data == True):
            height_shift = 0.1
            width_shift = 0.1

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=enhance_data, height_shift_range=height_shift, width_shift_range=width_shift)

        test_datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(DATASET_TRAIN_DIR, target_size=(training_image_size, training_image_size),
                                                            batch_size=batch_size,
                                                            class_mode="categorical")
        test_generator = test_datagen.flow_from_directory(DATASET_TEST_DIR, target_size=(training_image_size, training_image_size),
                                                          batch_size=batch_size,
                                                          class_mode="categorical")

        class_indices = train_generator.class_indices
        class_json = {}
        for eachClass in class_indices:
            class_json[str(class_indices[eachClass])] = eachClass

        with open(os.path.join(DATASET_Json_DIR, "model_class.json"), "w+") as json_file:
            json.dump(class_json, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)
            json_file.close()
        print("JSON Mapping for the model classes saved to ", os.path.join(DATASET_Json_DIR, "model_class.json"))

        num_train = len(train_generator.filenames)
        num_test = len(test_generator.filenames)
        print("Number of experiments (Epochs) : ", num_epochs)

        
        model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint,tensorboard,lr_scheduler])
      


