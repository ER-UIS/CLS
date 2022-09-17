from KerasModels.inception_resnet_v2  import InceptionResNetV2
from keras.preprocessing import image
import matplotlib.pyplot as plt

from KerasModels.custom_utils import decode_predictions, preprocess_input
from keras.layers import Input
from sklearn.metrics import roc_curve, auc
import csv
import os
import numpy as np
import pandas as pd
import glob
import time

def PredImage(picture,model,JSON_PATH,objects_num):
   
    image_to_predict = image.load_img(picture, target_size=(224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)

    image_to_predict = preprocess_input(image_to_predict)
    prediction = model.predict(x=image_to_predict, steps=1)
    
    predictiondata = decode_predictions(prediction, top=int(objects_num),model_json=JSON_PATH)
   
    return predictiondata

def mulimagezd(MODEL_PATH,JSON_PATH,model_roc,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type):
       
    model =InceptionResNetV2(weights=None, input_shape=(224, 224,3), classes=objects_num)
    
    model.load_weights(MODEL_PATH)
    pred=[]
    
    for files in glob.glob(os.path.join(DATASET_DIR, r"TDS\*.*")):
        filepath, filename = os.path.split(files)
        picture = os.path.join(DATASET_DIR, files)
        
        result=PredImage(picture,model,JSON_PATH,objects_num)
        
        if train_type==1:
            for s1 in  result:
                if s1[0]=="Malignant":
                    print(s1[1])
                    pred.append(s1[1])
        if train_type==2:
            for s1 in  result:
                if s1[0]=="invasive non-specialized carcinoma":
                    print(s1[1])
                    pred.append(s1[1])
        if train_type==3:
            for s1 in  result:
                if s1[0]=="invasive ductal carcinoma of the breast":
                    print(s1[1])
                    pred.append(s1[1])
        

    label = []
    with open('label400.csv','r', newline="",encoding="UTF-8") as csvfile:
        csv_reader = csv.reader(csvfile) 
        #birth_header = next(csv_reader)  
        for row in csv_reader:  
            label.append(row)

        label = [[int(x) for x in row] for row in label]   


    label=sum(label,[])   
    score=pred
    target=label
  
    fpr, tpr, thresholds = roc_curve(target, score)
    na = os.path.basename(MODEL_PATH)
    TestModel_data_name = na
    data=pd.DataFrame({'Classification':sum(label,[]),'diagnosis': pred})
    data.to_csv(DATASET_TestModel_timeDIR +r"\\" + TestModel_data_name+".csv", index=None, encoding='gbk')
    address1 = na.split('_')
    RUCN=address1[1]
    
    print(score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label= RUCN +"(auc = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    
    plt.legend(loc="lower right")
   
    TestModel_ROC = na
    
    plt.savefig(DATASET_TestModel_timeDIR +r"\\"+TestModel_ROC+".png",dpi=600)
    plt.savefig(DATASET_TestModel_timeDIR +r"\\"+TestModel_ROC+".pdf")
    
    model_roc.update({na:roc_auc})
     
    return  model_roc   
 




