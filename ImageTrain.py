import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from ERPrediction import mulimagezd
from ERTrain import trainModel,ERtrainModel
from KerasModels.inception_resnet_v2  import InceptionResNetV2

import warnings
from io import StringIO
import re
import collections
import os, random, shutil
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
import psutil


warnings.filterwarnings("ignore")

pathset=os.path.dirname(os.path.realpath(sys.argv[0]))
execution_path = os.getcwd()

  

#Determine if a program is running
def proc_exist(process_name):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process_name:
            return pid
#delete all files in the directory
def del_file(filepath):
   
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
#Get the number of all files in a directory           
def get_filenum(rootdir): 
    
    for root,dirs,files in os.walk(rootdir):
    
        for file in files:
            
            numfile=len(os.path.join(root,file))
    return  numfile

#get all files in a directory
def get_all(cwd):
    result = []
    get_dir = os.listdir(cwd)  

    for i in get_dir:          

        sub_dir = os.path.join(cwd,i)  

        if os.path.isdir(sub_dir):     

            get_all(sub_dir)

        else:

            result.append(i)
    
    return  result
#model optimization
model_roc={}
def modelTest(model_roc,MODELs_DIR,Json_DIR,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type):
    filel=get_all(os.path.join(execution_path, MODELs_DIR))
    if len(filel)>=8:
         filel=filel[:-9:-1]
        
    for file in filel:
        
        MODEL_PATH = os.path.join(MODELs_DIR, file)
        JSON_PATH = os.path.join(Json_DIR, "model_class.json")
        
        mulimagezd(MODEL_PATH,JSON_PATH,model_roc,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type)
        
    
    return max(model_roc.values()),max(model_roc,key=model_roc.get)
#Lump benign and malignant diagnosis model training
def ImageTrain():
 	
    Data_Path='model_tumor-breast'
    modelType=InceptionResNetV2
    ERtrainModel(Data_Path)
    DATASET_DIR = os.path.join(execution_path, Data_Path)
    MODELs_DIR = os.path.join(DATASET_DIR, "models")
    BestModel_DIR = os.path.join(DATASET_DIR, "BestModel")
    Json_DIR = os.path.join(DATASET_DIR, "json")
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)
    
    rootdir_train = pathset+ r'\model_tumor-breast\train'
    list = os.listdir(rootdir_train) 
    objects_num=len(list)
    train_type=1
    
    filenum=get_filenum(rootdir_train)
    if filenum<1000:
       num_exp=100
    else:
       num_exp=200 
    
    trainModel(modelType,num_objects=objects_num, num_experiments=num_exp, enhance_data=True,continue_from_model=None, transfer_from_model=True,batch_size=8, show_network_summary=True)
    #Selection of benign and malignant diagnostic models
    roc,file=modelTest(model_roc,MODELs_DIR,Json_DIR,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type)
    del_file(BestModel_DIR)
    shutil.copyfile(MODELs_DIR+r"\\"+file,BestModel_DIR + r"\\"+ "{:.3f}".format(roc)+"-"+file)
    
 #Lump pathological disease diagnosis model training          
def ImageTrainMd():
    
    Data_Path='model_Diseasel-breast'
    modelType=InceptionResNetV2
    ERtrainModel(Data_Path)
    DATASET_DIR = os.path.join(execution_path, Data_Path)
    MODELs_DIR = os.path.join(DATASET_DIR, "models")
    BestModel_DIR = os.path.join(DATASET_DIR, "BestModel")
    Json_DIR = os.path.join(DATASET_DIR, "json")
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)
    train_type=3
    
    rootdir_train = pathset+ r'\model_Diseasel-breast\train'
    list = os.listdir(rootdir_train) 
    objects_num=len(list)
    filenum=get_filenum(rootdir_train)
    if filenum<1000:
       num_exp=100
    else:
       num_exp=200 
    
    trainModel(modelType,num_objects=objects_num, num_experiments=20, enhance_data=True,continue_from_model=None, transfer_from_model=True,batch_size=8, show_network_summary=True)
    #The optimal model for diagnosis of mass pathological diseases
    roc,file=modelTest(model_roc,MODELs_DIR,Json_DIR,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type)
    del_file(BestModel_DIR)
    shutil.copyfile(MODELs_DIR+r"\\"+file,BestModel_DIR + r"\\"+ "{:.3f}".format(roc)+"-"+file)
       
#Model training for the diagnosis of tumor pathological types
def ImageTrainPMd():
  
    Data_Path='model_Pathology-breast'
    modelType=InceptionResNetV2
    ERtrainModel(Data_Path)
    DATASET_DIR = os.path.join(execution_path, Data_Path)
    MODELs_DIR = os.path.join(DATASET_DIR, "models")
    BestModel_DIR = os.path.join(DATASET_DIR, "BestModel")
    Json_DIR = os.path.join(DATASET_DIR, "json")
    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    DATASET_TestModel_DIR = os.path.join(DATASET_DIR, "TestModel")
    DATASET_TestModel_timeDIR = os.path.join(DATASET_TestModel_DIR, dir_time)
    train_type=2
    rootdir_train = pathset+ r'\model_Pathology-breast\train'
    list = os.listdir(rootdir_train) 
    objects_num=len(list)
    filenum=get_filenum(rootdir_train)
    if filenum<1000:
       num_exp=100
    else:
       num_exp=200 
    
    trainModel(modelType,num_objects=objects_num, num_experiments=num_exp, enhance_data=True,continue_from_model=None, transfer_from_model=True,batch_size=8, show_network_summary=True)
   #Optimization of the diagnostic model for the pathological type of the mass
    roc,file=modelTest(model_roc,MODELs_DIR,Json_DIR,DATASET_DIR,objects_num,DATASET_TestModel_timeDIR,train_type)
    del_file(BestModel_DIR)
    shutil.copyfile(MODELs_DIR+r"\\"+file,BestModel_DIR + r"\\"+ "{:.3f}".format(roc)+"-"+file)
   
 #Model update  
def Modelupdate():  
    
    if os.path.isdir('./modelbackups'):
        print("modelbackups directory exists!")
    else:
      
        os.mkdir('./modelbackups')

    if os.path.isdir('./modelbackups/breast'):
        print("modelbackups/breast directory exists!")
    else:
       
        os.mkdir('./modelbackups/breast')

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    
    if os.path.isdir('./modelbackups/breast/'+now):
        print("modelbackups/breast/current date directory exists!")
    else:
      
        os.mkdir('./modelbackups/breast/'+now)

    sourcePath = 'model_Diseasel-breast.h5'
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)
    sourcePath = 'model_Diseasel-breast_class.json'
    
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)

  
    sourcePath = 'model_Pathology-breast.h5'
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)
    sourcePath = 'model_Pathology-breast_class.json'
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)

    sourcePath = 'model_tumor-breast.h5'
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)
    sourcePath = 'model_tumor-breast_class.json'
    targetPath ='./modelbackups/breast/'+now
    if os.path.isfile(sourcePath):
       shutil.copy(sourcePath,targetPath)

   

    rootdir = pathset+ r'\model_tumor-breast\BestModel'
    list1 = os.listdir(rootdir) 
    print(list1[-1])
    sf=list1[-1]
    
    sourcefile=pathset+ './model_tumor-breast/BestModel/'+ sf
    targetfile='model_tumor-breast.h5'
   
    
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)
       
       
    os.rename(pathset+ './model_tumor-breast/BestModel/'+ sf, pathset+ './model_tumor-breast/BestModel/'+ "00_"+sf)
    sourcefile=pathset+ './model_tumor-breast/json/model_class.json'
    targetfile='model_tumor-breast_class.json'
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)

    rootdir = pathset+ r'\model_Diseasel-breast\BestModel'
    list1 = os.listdir(rootdir) 
    print(list1[-1])
    sf=list1[-1]
    sourcefile=pathset+ './model_Diseasel-breast/BestModel/'+ sf
    targetfile='model_Diseasel-breast.h5'
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)
    os.rename(pathset+ './model_Diseasel-breast/BestModel/'+ sf, pathset+ './model_Diseasel-breast/BestModel/'+ "00_"+sf)
    sourcefile=pathset+ './model_Diseasel-breast/json/model_class.json'
    targetfile='model_Diseasel-breast_class.json'
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)

    
    rootdir = pathset+ r'\model_Pathology-breast\BestModel'
    list1 = os.listdir(rootdir) 
    print(list1[-1])
    sf=list1[-1]
    sourcefile=pathset+ './model_Pathology-breast/BestModel/'+ sf
    targetfile='model_Pathology-breast.h5'
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)
    os.rename(pathset+ './model_Pathology-breast/BestModel/'+ sf, pathset+ './model_Pathology-breast/BestModel/'+ "00_"+sf)
    sourcefile=pathset+ './model_Pathology-breast/json/model_class.json'
    targetfile='model_Pathology-breast_class.json'
    if os.path.isfile(sourcefile):
       shutil.copy(sourcefile,targetfile)

def run(interval):
   
    while True:
         
        time_remaining = interval-time.time()%interval
        time.sleep(time_remaining)
        dbfile=pathset+"/OITDS.accdb"
        conn=pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};PWD=;DBQ='+dbfile)
        cur=conn.cursor()
        cur.execute("SELECT * FROM  patientdatatable where Image_diagnosis_status<>3 and Test_status=1")
        
        row=cur.fetchone()
        if row is not None:  
            if isinstance(proc_exist('ImageZdTest.exe'),int):
                    print('ImageZdTest.exe is running')
            else:
                print('no such ImageZdTest process...')
                fil = os.getcwd()+r"\ImageZdTest.exe"
            
                fil_run = os.system(fil)
                print(fil_run)
        cur.close()
        conn.close() 
        

if __name__=="__main__":
    ImageTrain()
    ImageTrainPMd()
    ImageTrainMd()
    Modelupdate()
    interval = 6
    run(interval)
   
   
   
    
        

