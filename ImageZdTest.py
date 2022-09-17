import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from KerasModels.inception_resnet_v2  import InceptionResNetV2
from ERPrediction import PredImage
import warnings
from io import StringIO
import re
import collections
import os
import sys
import time
import logging
import pyodbc
import pymssql
import decimal
import six
import packaging
import packaging.version
import packaging.specifiers
import packaging.requirements
import gc
import glob
import threading

warnings.filterwarnings("ignore")

pathset=os.path.dirname(os.path.realpath(sys.argv[0]))
dbfile=pathset+"/OITDS.accdb"
#Get part of a string
def get_str_btw(s, f, b):
    par = s.partition(b)
    print(par)
    return (par[0].rpartition(f))[2]           
def BreastImagezd():
   
    conn=pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};PWD=;DBQ='+dbfile)
    cur=conn.cursor()
    cur.execute("select patientID, Image_diagnosis_status,Pathological_diagnosis_image_path from patientdatatable where Test_status=1")
    
    row=cur.fetchone()
    
    if row is not None: 
        imglist=[]
        pID=row[0]
   
        Ids=row[1]
       
        imgljd=pathset+"/image/"+pID
       
        for each_file in os.listdir(imgljd):
            
             imglist.append(os.path.join(imgljd,each_file))
            
      
        imglist = ','.join(imglist)
     
                
        pdip=row[2]
       
        execution_path = os.getcwd()
        if Ids==0:
            model_ex= "model_tumor-breast.h5"
            model_class="model_tumor-breast_class.json"
            num_ob=2
            num_obp=2
            imglj=imglist
        if Ids==1:
           
            rootdir_train = pathset+ r'\model_Pathology-breast\train'
            list1 = os.listdir(rootdir_train) 
       
            model_ex= "model_Pathology-breast.h5"
            model_class="model_Pathology-breast_class.json"
            num_ob=len(list1)
            num_obp=3
            imglj=pdip
            
        
        if Ids==2:
            
            rootdir_train = pathset+ r'\model_Diseasel-breast\train'
            list1 = os.listdir(rootdir_train) 
           
            model_ex= "model_Diseasel-breast.h5"
            model_class="model_Diseasel-breast_class.json"
            num_ob=len(list1)
            
            num_obp=6
            imglj=pdip
        

        if imglj is not  None: 
            print("image path exists!")
            pred=[]
            prem=[]
           
            
            for lj in imglj.split(","):
                
                if os.path.exists(lj):
                    print(lj)
                    print("image exists!")
                    imlj=os.path.dirname(lj)
                  
                    print(imlj)
                    MODEL_PATH = os.path.join(execution_path, model_ex)
                    JSON_PATH = os.path.join(execution_path, model_class)
                    model =InceptionResNetV2(weights=None, input_shape=(224, 224,3), classes=num_ob)
                    model.load_weights(MODEL_PATH)
                                
                    picture = lj
                    predictionDict = {}
                    result=PredImage(picture,model,JSON_PATH,num_obp)
                    
                
                    for s1 in  result:
                        predictionDict[s1[0]] = s1[1]
                        if Ids==0:
                            mm= lj +"-"+ s1[0] +": "+ str("{:6f}".format(s1[1]*100))+"%"+","
                        else:
                            mm= s1[0] +": "+ str("{:6f}".format(s1[1]*100))+"%"+","
                                                
                        pred.append(mm)
                    
                    
                    PTD_P=max(predictionDict.values())
                    
                    PTD_r=max(predictionDict,key=predictionDict.get)
                        

                    PTD=str(" ".join(pred))
                    PDD_P=PTD_r +": "+ str(PTD_P)
                                  
                    
                    if Ids==0:
                        
                        s2=predictionDict["Malignant"]
                       
                        m2=float("{:6f}".format(s2*100))
                                              
                        
                        prem.append(m2)
                    
                    print("Image diagnosis OK")
            
            if Ids==0:
                pm=max(prem)
                
                Mdp=float("{:6f}".format(float(pm)/100))
               
             
                Icd=str(" ".join(pred))
                print(Icd)
                
                pm="{:6f}".format(pm)
                strpart="-Malignant: "+ str(pm)
                print(strpart)
                if Icd.find(strpart) >= 0:
                  
                    imfil=get_str_btw(Icd,"\\",strpart)
                    PDi_path= imlj + "\\"+ imfil  
                    
                    
                else:
                    pass
         
               
            if Ids==1:
                
                sql="update  patientdatatable set Image_diagnosis_status=2, PTD_Probability='%s', PTD_results='%s', Pathological_Type_Diagnosis='%s' where [patientID]='%s'" %((PTD_P,PTD_r,PTD,pID))
                
                
            
            if Ids==0:
                sql="update  patientdatatable set Image_diagnosis_status=1, Malignant_diagnosis_probability='%s',Image_classification_diagnosis='%s', Pathological_diagnosis_image_path='%s' where [patientID]='%s'" %((Mdp,Icd,PDi_path,pID))
            
            if Ids==2:
                sql="update  patientdatatable set Test_status=0, Image_diagnosis_status=3,  PDD_Probability='%s', Pathologica_disease_diagnosis='%s' where [patientID]='%s'" %((PDD_P,PTD,pID))
                         
            
            cur.execute(sql)
                      
            conn.commit()
            
            

        else:
         
            pass
    
    cur.close()
    conn.close()
def diagnostic_test():
    
        conn=pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};PWD=;DBQ='+dbfile)
        cur=conn.cursor()
        cur.execute("SELECT * FROM  patientdatatable where Image_diagnosis_status<>3 and Test_status=1")
        
        row=cur.fetchone()
      
        if row is not None:  
           
            thread = threading.Thread(target=BreastImagezd)
            thread.start()
        
    
        cur.close()
        conn.close() 
       
if __name__=="__main__":
  
    diagnostic_test()