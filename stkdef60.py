import warnings; warnings.simplefilter('ignore')
import logging
from kiteconnect import KiteConnect
logging.basicConfig(level=logging.ERROR)
import ssl
from requests.exceptions import ConnectionError
from http.client import RemoteDisconnected
from urllib3.exceptions import ProtocolError
from ssl import SSLError
import math
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, LSTM,Dropout,Input,Flatten,Activation,LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard 
import tensorflow.keras
import matplotlib.pyplot as plt
import nsepy
import pprint
import os
from datetime import date,timedelta
import datetime
plt.style.use('fivethirtyeight')
from tqdm import tqdm
import pickle
from sklearn import preprocessing
from xgboost import XGBClassifier
import xgboost as xgb
# import livelossplot
# from livelossplot import PlotLossesKeras
# from livelossplot.tf_keras import PlotLossesCallback
# from wandb.keras import WandbCallback
from keras_tqdm import TQDMNotebookCallback
import tensorflow as tf
from IPython.display import clear_output
import seaborn as sns
import cv2
import threading
import time
import glob
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
import ta
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# plt.style.use("seaborn")
homediraddr='/home/hemang'

class One_trian_stk_class:
    def __init__(self, stkname, insttkn,nbatch,nepochs,trainlen,kite):
        self.name = stkname
        self.insttkn = insttkn
        self.nepochs=nepochs
        self.nbatch=nbatch
        self.kite=kite
        self.trainlen=trainlen
        self.df=pd.DataFrame()
        self.scaler=MinMaxScaler(copy=True, feature_range=(0, 1))
        self.model=tf.keras.Model()
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.ta_x_train=[]
        self.ta_y_train=[]
        self.ta_x_test=[]
        self.ta_y_test=[]
        self.plotdir=''
        self.onestkscaledict={}
        self.scaled_data=np.array([])
        self.dscdata=np.array([])
        self.npmin=0
        self.npdiff=0
        self.modelpath=''
        self.talibdf=pd.DataFrame()
        self.dtrain = xgb.DMatrix
        self.dtest = xgb.DMatrix
        self.dxtrain=xgb.DMatrix
        self.dytrain=xgb.DMatrix
        self.dxtest=xgb.DMatrix
        self.dytest=xgb.DMatrix
        
        
    def load(self):
        # Load Stock Historical Data
        print('df loaded began................')
        filecheckpath =homediraddr+'/pfiles/'+self.name+'_'+'DF'+'.p'
        if(os.path.isfile(filecheckpath)):
            with open(filecheckpath, 'rb') as fp:
                if(self.trainlen=='all'):
                    self.df = pickle.load(fp)
                if(type(self.trainlen)==int):
                    self.df = pickle.load(fp).tail(self.trainlen)
                if(type(self.trainlen)==float):
                    newdff = pickle.load(fp)
                    trlen = math.ceil(len(newdff.index)*self.trainlen)
                    self.df = newdff.tail(trlen)
                
                
            print('Loaded : '+self.name+'_'+'DF'+'.p 1')
        else:
            old_lst=[]
            interval='minute'
            todaydt=datetime.date.today()
            hud_ago=todaydt-timedelta(days=59)
            to_date=datetime.date.isoformat(todaydt)
            from_date=datetime.date.isoformat(hud_ago)
            print('Getting Stock data : '+self.name)
            
            for i2 in tqdm(range(50)):
                clear_output(wait=True)
                new_lst = self.kite.historical_data(self.insttkn, from_date, to_date, interval,continuous=False)
                old_lst = new_lst + old_lst
                todaydt=todaydt-timedelta(days=60)
                hud_ago=hud_ago-timedelta(days=60)
                to_date=datetime.date.isoformat(todaydt)
                from_date=datetime.date.isoformat(hud_ago)
                print(len(old_lst))
                
            self.df=pd.DataFrame(old_lst)
            with open(homediraddr+'/pfiles/'+self.name+'_'+'DF'+'.p', 'wb') as fp:
                pickle.dump(self.df, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved file '+homediraddr+'/pfiles/'+self.name+'_'+'DF'+'.p 1')
            print('Loaded : '+self.name+'_'+'DF'+'.p 2')
        
        #set scaler
        newarr=self.df.drop(['date'], axis=1).values
        self.dscdata=newarr
        npmin=np.min(newarr)
        npmax=np.max(newarr)
        npdiff=npmax-npmin
        scaled_arr=(newarr-npmin)/npdiff
        
        self.npmin=npmin
        self.npdiff=npdiff
        self.scaled_data=scaled_arr

            
        # Load Model
        modeldir=homediraddr+ '/stkwise_models60/'+self.name
        filenum=0
        filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        while os.path.isfile(filecheckpath):
            filenum+=1
            filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        filecheckpath =modeldir+'/'+self.name+'_'+str(filenum-1)+'.h5'
        print('Checking Model Path : '+filecheckpath)
        if(os.path.isfile(filecheckpath)):
            self.model=load_model(filecheckpath)
            print('model loaded : '+filecheckpath)
            self.modelpath=filecheckpath
        elif(os.path.isfile(homediraddr+'/GRU_first_test.h5')):
            self.model=load_model(homediraddr+'/GRU_first_test.h5')
            print('model loaded : '+homediraddr+'/GRU_first_test.h5')
            self.modelpath=homediraddr+'/GRU_first_test.h5'
        else:
            print('model not loaded....')

        print('df loaded end................')


    def xtrain_def(self):
        print('xtrain_def began............')
        dataset=self.df.drop(['date'], axis=1).values
        print('dataset : '+str(dataset[0:5,:]))
        training_data_len = math.ceil( len(dataset) * .99 )
        print('training_data_len : '+str(training_data_len))
        print(self.scaled_data.shape)

        x_train = []
        y_train = []
        
        train_data = self.dscdata[0:training_data_len , :]
        
        for i in range(60, len(train_data)-61):
            x_train.append(train_data[i-60:i, :])
            y_train.append(train_data[i:i+60, 0])
        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        
        test_data = self.dscdata[training_data_len - 60: , :]
        x_test=[]
        y_test=[]
        for i in range(60, len(test_data)-61):
            x_test.append(test_data[i-60:i, :])
            y_test.append(test_data[i:i+60, 0])
        self.x_test, self.y_test = np.array(x_test), np.array(y_test)

        print(self.name +' : x_train shape : ',str(self.x_train.shape))
        print(self.name +' : y_train shape : ',str(self.y_train.shape))
        print(self.name +' : x_test shape : ',str(self.x_test.shape))
        print(self.name +' : y_test shape : ',str(self.y_test.shape))
        print('xtrain_def end............')
        
    def ta_xtrain_def(self):
        
#         op = self.df['open']
#         hi = self.df['high']
#         lo = self.df['low']
#         cl = self.df['close']
#         self.talibdf= self.df.drop(['date'],axis=1)
#         candle_names = talib.get_function_groups()['Pattern Recognition']
#         for candle in candle_names:
#             self.talibdf[candle] = getattr(talib, candle)(op, hi, lo, cl)
            
        df =self.df
        df= df.drop(['date'],axis=1)
        self.talibdf = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume").fillna(0)
    
        training_data_len = math.ceil( len(self.talibdf) * .99 )

        x_train = []
        y_train = []
        
        train_data = np.array(self.talibdf)[0:training_data_len , :]
        
        for i in range(60, len(train_data)-61):
            x_train.append(train_data[i-60:i, :])
            y_train.append(train_data[i:i+60, 0])
        self.ta_x_train, self.ta_y_train = np.array(x_train), np.array(y_train)

#         x_train=np.array(x_train)
#         x_train=x_train.reshape([x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
#         y_train=np.array(y_train)
#         self.dxtrain=xgb.DMatrix(x_train)
#         self.dytrain=xgb.DMatrix(y_train)
        
#         self.dtrain = xgb.DMatrix(x_train, label=y_train)
        ########################################################################################
        test_data = np.array(self.talibdf)[training_data_len - 60: , :]
        x_test=[]
        y_test=[]
        for i in range(60, len(test_data)-61):
            x_test.append(test_data[i-60:i, :])
            y_test.append(test_data[i:i+60, 0])
        self.ta_x_test, self.ta_y_test = np.array(x_test), np.array(y_test)
        
#         x_test=np.array(x_test)
#         x_test=x_test.reshape([x_test.shape[0],x_test.shape[1]*x_test.shape[2]])
#         y_test=np.array(y_test)
#         self.dxtest=xgb.DMatrix(x_test)
#         self.dytest=xgb.DMatrix(y_test)
#         self.dtest = xgb.DMatrix(x_test, label=y_test)
        
        print(self.name +' : ta_x_train shape : ',str(self.ta_x_train.shape))
        print(self.name +' : ta_y_train shape : ',str(self.ta_y_train.shape))
        print(self.name +' : ta_x_test shape : ',str(self.ta_x_test.shape))
        print(self.name +' : ta_y_test shape : ',str(self.ta_y_test.shape))
        print('ta_xtrain_def end............')
        

    def plot_predict(self):
        onemodelnameip=self.name
        new_df = self.df.drop(['date'], axis=1)
        scaler=self.onestkscaledict
        times=12
        oneintervalsize=60
#         print('plot predict began.............')
        for onetime in range(times):
            last_60_days = new_df[-oneintervalsize*(onetime+3):-oneintervalsize*(onetime+2)].values
            predicted_plot=last_60_days[:,0]
            actual_plot=new_df[-oneintervalsize*(onetime+3):-oneintervalsize*(onetime+1)].values[:,0]
#             last_60_days_scaled=(last_60_days-self.npmin)/self.npdiff
            last_60_days_scaled=last_60_days
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test=np.array(X_test)
            pred_price = self.model.predict(X_test)
#             pred_price_descaled=(pred_price*self.npdiff)+self.npmin
            pred_price_descaled = pred_price
            predicted_plot=np.append(predicted_plot,pred_price_descaled[0])
            plot_df=pd.DataFrame(predicted_plot.T)
            plot_df2=pd.DataFrame(actual_plot)
            result = pd.concat([plot_df, plot_df2], axis=1, sort=False)
            result.plot(figsize=(11,4));
            
            self.plotdir=homediraddr+ '/predict_plots2/'+self.name
            if(not os.path.isdir(self.plotdir)):
                os.mkdir(self.plotdir)
                print('Directory created : '+self.plotdir)
            filenum=0
            filecheckpath = self.plotdir + '/'+self.name+'_'+str(filenum)+'_'+str(onetime)+'.png'
            while os.path.isfile(filecheckpath):
                filenum+=1
                filecheckpath = self.plotdir + '/'+self.name+'_'+str(filenum)+'_'+str(onetime)+'.png'
            ax = plt.gca()
            plt.text(0.95, 0.01, self.modelpath,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='green', fontsize=15)
            
            plt.savefig(filecheckpath, dpi=100, bbox_inches='tight');

        f = []
        for (dirpath, dirnames, filenames) in os.walk(self.plotdir):
            f.extend(filenames)
            break
        itnum = math.floor(len(f)/times)
        firstcode=itnum-1
        if(len(glob.glob( self.plotdir + '/' + self.name + '_' + str(firstcode) + '*.png'))==times):
            for secondcode in range(times):
                globals()['a'+str(secondcode)]= cv2.imread(self.plotdir + '/'+self.name+'_'+str(firstcode)+'_'+str(secondcode)+'.png')
                globals()['b'+str(secondcode)] = cv2.resize(globals()['a'+str(secondcode)], dsize=(0, 0), fx=0.4, fy=0.4)
            def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                w_min = min(im.shape[1] for im in im_list)
                im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                                  for im in im_list]
                return cv2.vconcat(im_list_resize)

            def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                h_min = min(im.shape[0] for im in im_list)
                im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                                  for im in im_list]
                return cv2.hconcat(im_list_resize)

            def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
                im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
                return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
            im_tile = concat_tile_resize([[b0,b1,b2],
                                          [b3,b4,b5],
                                          [b6,b7,b8],
                                          [b9,b10,b11],])
            griddir=homediraddr+'/predict_plots2/'+self.name+'12'
            if(not os.path.isdir(griddir)):
                os.mkdir(griddir)
                print('Directory created : '+griddir)
            filenum=0
            filecheckpath = griddir+ '/'+self.name+'_'+str(filenum)+'.png'
            while os.path.isfile(filecheckpath):
                filenum+=1
                filecheckpath = self.plotdir+'12' + '/'+self.name+'_'+str(filenum)+'.png'
            cv2.imwrite(filecheckpath, im_tile)

    def savemodelh5(self):
        modeldir=homediraddr+ '/stkwise_models60/'+self.name
        if(not os.path.isdir(modeldir)):
            os.mkdir(modeldir)
            print('Directory created : '+modeldir)
        filenum=0
        filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        while os.path.isfile(filecheckpath):
            filenum+=1
            filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        self.model.save(modeldir+'/'+self.name+'_'+str(filenum)+'.h5')
        self.modelpath=modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        print('Model saved to : ' +self.modelpath)
        
    
    def ta_plot_predict(self):
        onemodelnameip=self.name
        new_df = self.df.drop(['date'], axis=1)
        scaler=self.onestkscaledict
        times=12
        oneintervalsize=60
#         print('plot predict began.............')
        for onetime in range(times):
            last_60_days = new_df[-oneintervalsize*(onetime+3):-oneintervalsize*(onetime+2)].values
            predicted_plot=last_60_days[:,0]
            actual_plot=new_df[-oneintervalsize*(onetime+3):-oneintervalsize*(onetime+1)].values[:,0]
            last_60_days_scaled=(last_60_days-self.npmin)/self.npdiff
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test=np.array(X_test)
            pred_price = self.model.predict(X_test)
            pred_price_descaled=(pred_price*self.npdiff)+self.npmin
            
            predicted_plot=np.append(predicted_plot,pred_price_descaled[0])
            plot_df=pd.DataFrame(predicted_plot.T)
            plot_df2=pd.DataFrame(actual_plot)
            result = pd.concat([plot_df, plot_df2], axis=1, sort=False)
            result.plot(figsize=(11,4));
            
            self.plotdir=homediraddr+ '/predict_plots2/'+self.name
            if(not os.path.isdir(self.plotdir)):
                os.mkdir(self.plotdir)
                print('Directory created : '+self.plotdir)
            filenum=0
            filecheckpath = self.plotdir + '/'+self.name+'_'+str(filenum)+'_'+str(onetime)+'.png'
            while os.path.isfile(filecheckpath):
                filenum+=1
                filecheckpath = self.plotdir + '/'+self.name+'_'+str(filenum)+'_'+str(onetime)+'.png'
            ax = plt.gca()
            plt.text(0.95, 0.01, self.modelpath,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='green', fontsize=15)
            
            plt.savefig(filecheckpath, dpi=100, bbox_inches='tight');

        f = []
        for (dirpath, dirnames, filenames) in os.walk(self.plotdir):
            f.extend(filenames)
            break
        itnum = math.floor(len(f)/times)
        firstcode=itnum-1
        if(len(glob.glob( self.plotdir + '/' + self.name + '_' + str(firstcode) + '*.png'))==times):
            for secondcode in range(times):
                globals()['a'+str(secondcode)]= cv2.imread(self.plotdir + '/'+self.name+'_'+str(firstcode)+'_'+str(secondcode)+'.png')
                globals()['b'+str(secondcode)] = cv2.resize(globals()['a'+str(secondcode)], dsize=(0, 0), fx=0.4, fy=0.4)
            def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                w_min = min(im.shape[1] for im in im_list)
                im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                                  for im in im_list]
                return cv2.vconcat(im_list_resize)

            def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                h_min = min(im.shape[0] for im in im_list)
                im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                                  for im in im_list]
                return cv2.hconcat(im_list_resize)

            def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
                im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
                return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
            im_tile = concat_tile_resize([[b0,b1,b2],
                                          [b3,b4,b5],
                                          [b6,b7,b8],
                                          [b9,b10,b11],])
            griddir=homediraddr+'/predict_plots2/'+self.name+'12'
            if(not os.path.isdir(griddir)):
                os.mkdir(griddir)
                print('Directory created : '+griddir)
            filenum=0
            filecheckpath = griddir+ '/'+self.name+'_'+str(filenum)+'.png'
            while os.path.isfile(filecheckpath):
                filenum+=1
                filecheckpath = self.plotdir+'12' + '/'+self.name+'_'+str(filenum)+'.png'
            cv2.imwrite(filecheckpath, im_tile)

    def savemodelh5(self):
        modeldir=homediraddr+ '/stkwise_models60/'+self.name
        if(not os.path.isdir(modeldir)):
            os.mkdir(modeldir)
            print('Directory created : '+modeldir)
        filenum=0
        filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        while os.path.isfile(filecheckpath):
            filenum+=1
            filecheckpath =modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        self.model.save(modeldir+'/'+self.name+'_'+str(filenum)+'.h5')
        self.modelpath=modeldir+'/'+self.name+'_'+str(filenum)+'.h5'
        print('Model saved to : ' +self.modelpath)
    
        
        
    def loadfit(self):
        self.load()
        self.xtrain_def()
#         self.plot_predict()
        x = threading.Thread(target=self.plot_predict, args=())
        x.start()
#         x.join()
        history=self.model.fit(
                          self.x_train,
                          self.y_train,
                          verbose=1,
                          shuffle=False,
                          batch_size=self.nbatch, 
                          epochs=self.nepochs,
                          validation_data=(self.x_test, self.y_test),
                          callbacks=[
#                                      PlotLossesKeras(),
                                     TrainingPlot(),
                                     MyCustomCallback(self),
                                    ])
        y = threading.Thread(target=self.savemodelh5, args=())
        y.start()
        
    def fit(self):
        history=self.model.fit(
                          self.x_train,
                          self.y_train,
                          verbose=1,
                          batch_size=self.nbatch, 
                          epochs=self.nepochs,
                          shuffle=False,
                          validation_data=(self.x_test, self.y_test),
                          callbacks=[
#                                      PlotLossesKeras(),
                                     TrainingPlot(),
                                     MyCustomCallback(self),
                                    ])


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, ipobj):
        self.ipobj = ipobj
    
    counter1=0
    def on_train_batch_begin(self, batch, logs=None):
        global counter1
        counter1+=1
        
        if(counter1%500==0):
            x = threading.Thread(target=self.ipobj.plot_predict, args=())
            x.start()
        
        if(counter1%2500==0):
            x = threading.Thread(target=self.ipobj.savemodelh5, args=())
            x.start()
        
#     def on_train_batch_end(self, batch, logs=None):
#         clear_output(wait=True)   
           
counter1=0


class TrainingPlot(tf.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure();
            plt.plot(N, self.losses, label = "train_loss");
            plt.plot(N, self.val_losses, label = "val_loss");
            plt.title("Training Loss");
            plt.xlabel("Epoch #");
            plt.ylabel("Loss");
            plt.legend();
            plt.savefig('losses.png');
            
            plt.figure();
            plt.plot(N, self.acc, label = "train_acc");
            plt.plot(N, self.val_acc, label = "val_acc");
            plt.title("Training Accuracy");
            plt.xlabel("Epoch #");
            plt.ylabel("Accuracy");
            plt.legend();
            plt.savefig('Accuracy.png');