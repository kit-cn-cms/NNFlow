from __future__ import absolute_import, division, print_function

import sys
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ROOT
ROOT.gROOT.SetBatch(1)

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score

from NNFlow.data_frame.data_frame import DataFrame              as NNFlowDataFrame


class NeuralNetworkTrainer(object):
    def __init__(self, nEpoch_class,nEpoch_adv,nEpoch_total):

        # Limit gpu usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        self.class_Epoch = nEpoch_class
        self.adv_Epoch = nEpoch_adv
        self.total_Epoch = nEpoch_total

    def train(self,
              save_path,
              model_id,
              hidden_layers,
              activation_function_name,
              dropout_keep_probability,
              l2_regularization_beta,
              early_stopping_intervall,
              path_to_training_data_set,
              path_to_validation_data_set,
              path_to_test_data_set,
              batch_size_training,
              ):

        print('\n' + '=======================')
        print('TRAINING NEURAL NETWORK')
        print('=======================' + '\n')

        # Check given paths
        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")

        if not os.path.isfile(path_to_training_data_set):
            sys.exit("File '" + path_to_training_data_set + "' doesn't exist." + "\n")

        if not os.path.isfile(path_to_validation_data_set):
            sys.exit("File '" + path_to_validation_data_set + "' doesn't exist." + "\n")

        directory_model_properties = os.path.join(save_path, 'model_properties')
        directory_plots = os.path.join(save_path, 'plots')

        # Create directories
        if not os.path.isdir(directory_model_properties):
            os.mkdir(directory_model_properties)
        if not os.path.isdir(directory_plots):
            os.mkdir(directory_plots)

        # Load data.
        training_data_set = NNFlowDataFrame(path_to_training_data_set)
        validation_data_set = NNFlowDataFrame(path_to_validation_data_set)
        test_data_set = NNFlowDataFrame(path_to_test_data_set)

        # Drop the given parameter in the dataset

        number_of_input_neurons = training_data_set.get_number_of_input_neurons()
        number_of_output_neurons = training_data_set.get_number_of_output_neurons()

        # Get weights and labels
        train_label_tmp, train_weights = training_data_set.get_labels_event_weights()
        vali_label_tmp, vali_weights = validation_data_set.get_labels_event_weights()
        test_label_tmp, test_weights = test_data_set.get_labels_event_weights()

        train_weights_adv = np.array(list(train_weights))
        vali_weights_adv = np.array(list(vali_weights))

        # Powheg tth
        train_label_class= [item[2] for item in train_label_tmp]
        vali_label_class = [item[2] for item in vali_label_tmp]
        test_label_class = [item[2] for item in test_label_tmp]
        

        train_label= [item[1] for item in train_label_tmp]
        vali_label = [item[1] for item in vali_label_tmp]
        test_label = [item[1] for item in test_label_tmp]

        # SHERPA 
        _train_label= [item[0] for item in train_label_tmp]
        _vali_label = [item[0] for item in vali_label_tmp]
        _test_label = [item[0] for item in test_label_tmp]

        for i,entry in enumerate(train_label_class):
            if entry==1:
                train_weights_adv[i]=0

        for i,entry in enumerate(vali_label_class):
            if entry==1:
                vali_weights_adv[i]=0

        for i,entry in enumerate(_train_label):
            if entry==1:
                train_weights[i]=0

        for i,entry in enumerate(_vali_label):
            if entry==1:
                vali_weights[i]=0

        '''
        train_label = [sum(x) for x in zip(train_label,train_label_class)]
        vali_label = [sum(x) for x in zip(vali_label,vali_label_class)]
        test_label = [sum(x) for x in zip(test_label,test_label_class)]
        '''

        # Get scaled data. data is scaled between 0 and 1
        train_data = training_data_set.get_scaled_data()
        vali_data = validation_data_set.get_scaled_data()
        test_data = test_data_set.get_scaled_data()

        '''
        label = np.expand_dims(np.array(train_label_class), axis=1)
        train_data=np.append(train_data,label,axis=1)

        label = np.expand_dims(np.array(vali_label_class), axis=1)
        vali_data=np.append(vali_data,label,axis=1)
        '''


        '''for i, ii in enumerate(train_data):
            for j, _ in enumerate(ii):
                train_data[i][j]= np.random.random()
                #vali_data[i][j]=np.random.random()
        '''


        # Build the model with the given parameters
        Inputs= keras.layers.Input(shape=(number_of_input_neurons,))
        X= Inputs
        if dropout_keep_probability != 1:
            X = keras.layers.Dropout(dropout_keep_probability)(X)
        for i in range(len(hidden_layers) - 1):
            X =keras.layers.Dense(hidden_layers[i],
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)
            if dropout_keep_probability != 1:
                X= keras.layers.Dropout(dropout_keep_probability)(X)
        # Build last layer, regression only works with 1 parameter
        X= keras.layers.Dense(1,
                                         activation='sigmoid',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)

        class_model = keras.models.Model(inputs=[Inputs], outputs=[X])

        adv_layers = class_model(Inputs)
        adv_layers = keras.layers.Dense(100,
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_layers = keras.layers.Dense(100,
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_layers = keras.layers.Dense(1,
                                activation='sigmoid',
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_model = keras.models.Model(inputs=[Inputs], outputs = [adv_layers])

        # Create the optimier. Here the Adamoptimizer is used but can be changed to a different optimizer
        optimizer = tf.train.AdamOptimizer(1e-5)

        def make_loss_Class(c):
            def loss_Class(y_true, y_pred):
                return c*keras.losses.binary_crossentropy(y_true,y_pred)
            return loss_Class

        def make_loss_Adv(c):
            def loss_Adv(z_true, z_pred):
                return c * keras.losses.binary_crossentropy(z_true,z_pred)

            return loss_Adv

        class_model.compile(loss=[make_loss_Class(c=1.0)],optimizer=optimizer)
        class_model.summary()

        adv_model.compile(loss=[make_loss_Adv(c=1.0)], optimizer=optimizer)
        adv_model.summary()

        # Compile the model with mean squared error as the loss function
        adv_model.trainable = False
        class_model.trainable = True
        class_adv_model = keras.models.Model(inputs=[Inputs], outputs=[class_model(Inputs),adv_model(Inputs)])
        class_adv_model.compile(loss=[make_loss_Class(c=1.0),make_loss_Adv(c=-50)],
                      optimizer=optimizer)

        class_adv_model.summary()
        adv_model.trainable = True
        class_model.trainable = False
        adv_class_model =keras.models.Model(inputs=[Inputs], outputs=[adv_model(Inputs)])
        adv_class_model.compile(loss =[make_loss_Adv(c=1.)], optimizer=optimizer)

        adv_class_model.summary()
        # Define earlystopping to reduce overfitting
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        ival = IntervalEvaluation(validation_data=(vali_data, vali_label), interval=1)
        # time_callback is used to get the time per epoch
        time_callback = TimeHistory()

        # Define max number of epochs
        nEpochs = 200
        # Train the NN
        #Pretrain networks
        ival = IntervalEvaluation(validation_data=(vali_data, vali_label), interval=1)
        
        adv_model.trainable = False
        class_model.trainable = True
        class_model.fit(x=train_data,
                        y=train_label_class,
                        verbose=1,
                        epochs=20,
                        batch_size=500,
                        validation_data=(vali_data, vali_label, vali_weights),
                        sample_weight=train_weights)
        


        y_pred = class_model.predict(train_data, verbose=1)

        score = roc_auc_score(train_label_class, y_pred)
        print('##############################################################################')
        print(score)
        print('##############################################################################')

        tth_sig = []
        ttbb_sherpa = []
        ttbb_powheg = []
        for i,entry in  enumerate(train_label_tmp):
            for j,n in enumerate(entry):
                if n==1:
                    if j==0:
                        ttbb_sherpa.append(y_pred[i][0])
                    elif j==1:
                        ttbb_powheg.append(y_pred[i][0])
                    elif j==2:
                        tth_sig.append(y_pred[i][0])

        plt.clf()

        bin_edges = np.linspace(0, 1, 30)

        plt.hist(tth_sig,     bins=bin_edges,     histtype='step', lw=1.5, label='ttH Powheg', normed='True', color='#1f77b4')
        plt.hist(ttbb_sherpa, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Sherpa', normed='True', color='#d62728')
        plt.hist(ttbb_powheg, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Powheg', normed='True', color='#008000')
        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')
        plt.title("Before Adversary Training")
        plt.savefig(os.path.join(directory_model_properties,'Signal_BKg_plot_before.pdf'))

        plt.clf()

        chi_h1 = ROOT.TH1F("Sherpa_chi2","Sherpa_chi2",20,-0.01,1.01)
        chi_h1.Sumw2()
        for entry in ttbb_sherpa:
            chi_h1.Fill(entry)
        chi_h1.Scale(1.0/chi_h1.Integral())
        chi_h2 = ROOT.TH1F("Powheg_chi2","Powheg_chi2",20,-0.01,1.01)
        chi_h2.Sumw2()
        for entry in ttbb_powheg:
            chi_h2.Fill(entry)
        chi_h2.Scale(1.0/chi_h2.Integral()) 
        canvas = ROOT.TCanvas("canvas","canvas",800,900)
        chi_h1.Draw("Hist")
        chi_h2.Draw("HISTSAME")
        canvas.SaveAs(os.path.join(directory_model_properties,'TestPlot'+".png"))
        print("Chi2 TEST")
        chi_h1.Chi2Test(chi_h2,"WUP")
        print("Kolmogorov:")
        print(chi_h1.KolmogorovTest(chi_h2,"ND"))
        
        adv_model.trainable = True
        class_model.trainable = False
        adv_class_model.fit(x=train_data,
                            y=train_label,
                            verbose=1,
                            epochs=20,
                            batch_size = 500,
                            validation_data=(vali_data, vali_label, vali_weights_adv),
                            sample_weight=train_weights_adv)
        

        loss_list = ['loss', 'model_1_loss',  'model_loss' ]
        loss_dict = {}
        adv_loss = {}

        adv_predict_train = adv_class_model.predict(train_data,batch_size=500,verbose=1)
        adv_predict_val = adv_class_model.predict(vali_data,batch_size=500,verbose=1)
        class_predict_train = class_model.predict(train_data,batch_size=500,verbose=1)
        class_predict_val = class_model.predict(vali_data,batch_size=500,verbose=1)
        maxX = -1000
        minX = 1000
        some_dict={}
        maxX = max(maxX,max(adv_predict_train))
        minX = min(minX,min(adv_predict_train))
        maxX = max(maxX,max(adv_predict_val))
        minX = min(minX,min(adv_predict_val))
        maxX = max(maxX,max(train_label))
        minX = min(minX,min(train_label))
        value_dict = {'adversary_prediction_training': adv_predict_train,
                      'adversary_prediction_valiadation':adv_predict_val,
        }

        h2= ROOT.TH1F("hist_label",'bla',20,minX,maxX)
        h2.SetLineColor(2)
        for entry in train_label:
            h2.Fill(entry)
        for key, entry in value_dict.iteritems():
            h1 = ROOT.TH1F("hist"+key,key,20,minX-0.01,maxX+0.01)
            for j in entry:
                h1.Fill(j)
            canvas=ROOT.TCanvas("canvas"+key,"canvas"+key,800,900)
            h1.SetLineColor(1)
            h1.Draw("HIST")
            h2.Draw("HISTSAME")
            canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_before_'+str(1)+"_"+key+".png"))
        h3 = ROOT.TH1F('class','class_output',20,-0.01,1.01)
        for i in class_predict_train:
            h3.Fill(i)
        canvas = ROOT.TCanvas("canvas","canvas",800,900)
        h3.Draw("Hist")
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Class_output_before_adv'+".png"))

        h1_2D = ROOT.TH2F('h1', 'Input_class_vs_outpu_class', 1000,min(train_label)-0.01,max(train_label)+0.01,1000,min(y_pred)-0.01,max(y_pred)+0.01)
        for i in range(len(train_label)):
            h1_2D.Fill(train_label[i],y_pred[i])
        print("\n")
        print('Input_class_vs_outpu_class')
        print(h1_2D.GetCorrelationFactor())

        h2_2D = ROOT.TH2F('h2', 'Input_class_vs_adv_output', 1000,min(train_label)-0.01,max(train_label)+0.01,1000,min(adv_predict_train)-0.01,max(adv_predict_train)+0.01)
        
        for i in range(len(train_label)):
            h2_2D.Fill(train_label[i],adv_predict_train[i])
        print("\n")
        print('Input_class_vs_adv_output')
        print(h2_2D.GetCorrelationFactor())

        h3_2D = ROOT.TH2F('h2', 'output_class_vs_adv_output', 1000,min(y_pred)-0.01,max(y_pred)+0.01,1000,min(adv_predict_train)-0.01,max(adv_predict_train)+0.01)
        
        for i in range(len(train_label)):
            h3_2D.Fill(y_pred[i],adv_predict_train[i])
        print("\n")
        print('output_class_vs_adv_output')
        print(h3_2D.GetCorrelationFactor())

        print("\n")
        print("\n")
        canvas=ROOT.TCanvas("canvas1","canvas1",800,900)
        h1_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_before'+str(1)+"_"+".png"))

        canvas=ROOT.TCanvas("canvas2","canvas2",800,900)
        h2_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_before'+str(2)+"_"+".png")) 

        canvas=ROOT.TCanvas("canvas3","canvas3",800,900)
        h3_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_before'+str(3)+"_"+".png")) 
        some_dict['Evt_blr_ETH']= [h1_2D.GetCorrelationFactor(),h2_2D.GetCorrelationFactor(),h3_2D.GetCorrelationFactor()]
        print(some_dict)


        for entry in loss_list:
            loss_dict[entry]=[]

        for entry in ['loss','val_loss']:
            adv_loss[entry]= []
        h=0
        for i in range(self.total_Epoch):
            adv_model.trainable = False
            class_model.trainable = True
            history =class_adv_model.fit(x= train_data,
                                y=[train_label_class,train_label],
                                verbose=1,
                                epochs=self.class_Epoch,
                                batch_size=500,
                                callbacks=[time_callback],
                                validation_data=(vali_data, [vali_label_class,vali_label],[vali_weights,vali_weights_adv]),
                                sample_weight=[train_weights,train_weights_adv]
                                )
            y_pred = class_model.predict(train_data, verbose=1)
            ttbb_sherpa = []
            ttbb_powheg = []
            for i,entry in  enumerate(train_label_tmp):
                for j,n in enumerate(entry):
                    if n==1:
                        if j==0:
                            ttbb_sherpa.append(y_pred[i][0])
                        elif j==1:
                            ttbb_powheg.append(y_pred[i][0])
                        elif j==2:
                            tth_sig.append(y_pred[i][0])

            plt.clf()

            bin_edges = np.linspace(0, 1, 30)

            plt.hist(tth_sig,     bins=bin_edges,     histtype='step', lw=1.5, label='ttH', normed='True', color='#1f77b4')
            plt.hist(ttbb_sherpa, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Sherpa', normed='True', color='#d62728')
            plt.hist(ttbb_powheg, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Powheg', normed='True', color='#008000')
            print(len(ttbb_sherpa))
            print(len(ttbb_powheg))
            print(len(tth_sig))
            plt.legend(loc='upper left')
            plt.xlabel('Network Output')
            plt.ylabel('Events (normalized)')
            plt.title("Before Adversary Training")
            plt.savefig(os.path.join(directory_model_properties,'Before_Adversary_Epoch: '+str(h)+'.png'))
            chi_h1 = ROOT.TH1F("Sherpa_chi2","Sherpa_chi2",20,-0.01,1.01)
            chi_h1.Sumw2()
            for entry in ttbb_sherpa:
                chi_h1.Fill(entry)
            chi_h1.Scale(1.0/chi_h1.Integral())
            chi_h2 = ROOT.TH1F("Powheg_chi2","Powheg_chi2",20,-0.01,1.01)
            chi_h2.Sumw2()
            for entry in ttbb_powheg:
                chi_h2.Fill(entry)
            chi_h2.Scale(1.0/chi_h2.Integral()) 
            canvas = ROOT.TCanvas("canvas","canvas",800,900)
            chi_h1.Draw("Hist")
            chi_h2.Draw("HISTSAME")
            canvas.SaveAs(os.path.join(directory_model_properties,'TestPlot'+".png"))
            print("\n")
            print("Chi2 TEST")
            chi_h1.Chi2Test(chi_h2,"WUP")
            print("Kolmogorov:")
            print(chi_h1.KolmogorovTest(chi_h2))
            print("\n")
            plt.clf()



            for entry in loss_list:
                loss_dict[entry].append(history.history[entry])
            adv_model.trainable = True
            class_model.trainable = False
            history = adv_class_model.fit(x=train_data,
                                y=train_label,
                                verbose=1,
                                epochs=self.adv_Epoch,
                                batch_size=500,
                                callbacks=[time_callback],
                                validation_data=(vali_data, vali_label, vali_weights_adv),
                                sample_weight=train_weights_adv)
            y_pred = adv_model.predict(train_data, verbose=1)
            ttbb_sherpa = []
            ttbb_powheg = []
            tth_sig =[]
            for i,entry in  enumerate(train_label_tmp):
                for j,n in enumerate(entry):
                    if n==1:
                        if j==0:
                            ttbb_sherpa.append(y_pred[i][0])
                        elif j==1:
                            ttbb_powheg.append(y_pred[i][0])

            plt.clf()

            bin_edges = np.linspace(0, 1, 30)

            plt.hist(ttbb_sherpa, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Sherpa', normed='True', color='#d62728')
            plt.hist(ttbb_powheg, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Powheg', normed='True', color='#008000')

            plt.legend(loc='upper left')
            plt.xlabel('Network Output')
            plt.ylabel('Events (normalized)')
            plt.title("After Adversary Training")
            plt.savefig(os.path.join(directory_model_properties,'After_Adversary_Epoch: '+str(h)+'.png'))

            plt.clf()
            chi_h1 = ROOT.TH1F("Sherpa_chi2","Sherpa_chi2",20,-0.01,1.01)
            chi_h1.Sumw2()
            for entry in ttbb_sherpa:
                chi_h1.Fill(entry)
            chi_h1.Scale(1.0/chi_h1.Integral())
            chi_h2 = ROOT.TH1F("Powheg_chi2","Powheg_chi2",20,-0.01,1.01)
            chi_h2.Sumw2()
            for entry in ttbb_powheg:
                chi_h2.Fill(entry)
            chi_h2.Scale(1.0/chi_h2.Integral()) 
            canvas = ROOT.TCanvas("canvas","canvas",800,900)
            chi_h1.Draw("Hist")
            chi_h2.Draw("HISTSAME")
            canvas.SaveAs(os.path.join(directory_model_properties,'TestPlot'+".png"))
            #print("Chi2 TEST")
            #chi_h1.Chi2Test(chi_h2,"WUP")
            #print("Kolmogorov:")
            #print(chi_h1.KolmogorovTest(chi_h2))


            h+=1

            for key in adv_loss:
                adv_loss[key].append(history.history[key])
        y_pred = adv_model.predict(train_data, verbose=1)
        ttbb_sherpa = []
        ttbb_powheg = []
        tth_sig =[]
        for i,entry in  enumerate(train_label_tmp):
            for j,n in enumerate(entry):
                if n==1:
                    if j==0:
                        ttbb_sherpa.append(y_pred[i][0])
                    elif j==1:
                        ttbb_powheg.append(y_pred[i][0])

        plt.clf()

        bin_edges = np.linspace(0, 1, 30)

        plt.hist(ttbb_sherpa, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Sherpa', normed='True', color='#d62728')
        plt.hist(ttbb_powheg, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Powheg', normed='True', color='#008000')
        print(len(ttbb_sherpa))
        print(len(ttbb_powheg))
        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')
        plt.title("After Adversary Training")
        plt.savefig(os.path.join(directory_model_properties,'After_Adversary_Epoch:50.png'))

        plt.clf()
        chi_h1 = ROOT.TH1F("Sherpa_chi2","Sherpa_chi2",20,-0.01,1.01)
        chi_h1.Sumw2()
        for entry in ttbb_sherpa:
            chi_h1.Fill(entry)
        chi_h1.Scale(1.0/chi_h1.Integral())
        chi_h2 = ROOT.TH1F("Powheg_chi2","Powheg_chi2",20,-0.01,1.01)
        chi_h2.Sumw2()
        for entry in ttbb_powheg:
            chi_h2.Fill(entry)
        chi_h2.Scale(1.0/chi_h2.Integral()) 
        canvas = ROOT.TCanvas("canvas","canvas",800,900)
        chi_h1.Draw("Hist")
        chi_h2.Draw("HISTSAME")
        canvas.SaveAs(os.path.join(directory_model_properties,'TestPlot'+".png"))
        print("Chi2 TEST")
        chi_h1.Chi2Test(chi_h2,"WUP")
        print("Kolmogorov:")
        print(chi_h1.KolmogorovTest(chi_h2))
        for key in loss_list:
            plt.clf()
            plt.plot(loss_dict[key],label=key)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(directory_model_properties,'Loss_Classifier.png'))

        for key in adv_loss:
            plt.clf()
            plt.plot(adv_loss[key],label=key)
        plt.legend()
        plt.savefig(os.path.join(directory_model_properties,'Loss_Adv.png'))

        adv_predict_train = adv_class_model.predict(train_data,batch_size=500,verbose=1)
        adv_predict_val = adv_class_model.predict(vali_data,batch_size=500,verbose=1)
        class_predict_train = class_model.predict(train_data,batch_size=500,verbose=1)
        class_predict_val = class_model.predict(vali_data,batch_size=500,verbose=1)
        maxX = -1000
        minX = 1000
        some_dict={}
        maxX = max(maxX,max(adv_predict_train))
        minX = min(minX,min(adv_predict_train))
        maxX = max(maxX,max(adv_predict_val))
        minX = min(minX,min(adv_predict_val))
        maxX = max(maxX,max(train_label))
        minX = min(minX,min(train_label))
        value_dict = {'adversary_prediction_training': adv_predict_train,
                      'adversary_prediction_valiadation':adv_predict_val,
        }

        h2= ROOT.TH1F("hist_label",'bla',20,minX,maxX)
        h2.SetLineColor(2)
        for entry in train_label:
            h2.Fill(entry)
        for key, entry in value_dict.iteritems():
            h1 = ROOT.TH1F("hist"+key,key,20,minX-0.01,maxX+0.01)
            for j in entry:
                h1.Fill(j)
            canvas=ROOT.TCanvas("canvas"+key,"canvas"+key,800,900)
            h1.SetLineColor(1)
            h1.Draw("HIST")
            h2.Draw("HISTSAME")
            canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_'+str(1)+"_"+key+".png"))
        h3 = ROOT.TH1F('class','class_output',20,-0.01,1.01)
        for i in class_predict_train:
            h3.Fill(i)
        canvas = ROOT.TCanvas("canvas","canvas",800,900)
        h3.Draw("Hist")
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Class_output_'+".png"))

        h1_2D = ROOT.TH2F('h1', 'Input_class_vs_output_class', 1000,min(train_label)-0.01,max(train_label)+0.01,1000,min(class_predict_train)-0.01,max(class_predict_train)+0.01)
        for i in range(len(train_label)):
            h1_2D.Fill(train_label[i],class_predict_train[i])
        print("\n")
        print('Input_class_vs_outpu_class')
        print(h1_2D.GetCorrelationFactor())

        h2_2D = ROOT.TH2F('h2', 'Input_class_vs_adv_output', 1000,min(train_label)-0.01,max(train_label)+0.01,1000,min(adv_predict_train)-0.01,max(adv_predict_train)+0.01)
        
        for i in range(len(train_label)):
            h2_2D.Fill(train_label[i],adv_predict_train[i])
        print("\n")
        print('Input_class_vs_adv_output')
        print(h2_2D.GetCorrelationFactor())

        h3_2D = ROOT.TH2F('h2', 'output_class_vs_adv_output', 1000,min(class_predict_train)-0.01,max(class_predict_train)+0.01,1000,min(adv_predict_train)-0.01,max(adv_predict_train)+0.01)
        
        for i in range(len(train_label)):
            h3_2D.Fill(class_predict_train[i],adv_predict_train[i])
        print("\n")
        print('output_class_vs_adv_output')
        print(h3_2D.GetCorrelationFactor())

        print("\n")
        print("\n")
        canvas=ROOT.TCanvas("canvas1","canvas1",800,900)
        h1_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_after_adv'+str(1)+"_"+".png"))

        canvas=ROOT.TCanvas("canvas2","canvas2",800,900)
        h2_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_after_adv'+str(2)+"_"+".png")) 

        canvas=ROOT.TCanvas("canvas3","canvas3",800,900)
        h3_2D.Draw('COLZ')  
        canvas.SaveAs(os.path.join(directory_model_properties,'ROOT_Plot_Epoch_True_2D_after_adv'+str(3)+"_"+".png")) 
        some_dict['Evt_blr_ETH']= [h1_2D.GetCorrelationFactor(),h2_2D.GetCorrelationFactor(),h3_2D.GetCorrelationFactor()]
        print(some_dict)


        # Get the time spend per epoch
        times = time_callback.times
        # Determine the number of epochs the NN trained
        if earlyStopping.stopped_epoch == 0:
            Epoch_end = nEpochs
        else:
            Epoch_end = earlyStopping.stopped_epoch


        # Plot the history of the loss function
        # self._plot_history(history= history,path=directory_plots)
        print('\n')
        print("Training finished, evaluating model on test data set")
        print('\n')
        # Evaluate the trained model on the test data sample
        y_pred = class_model.predict(train_data, verbose=1)

        score = roc_auc_score(train_label_class, y_pred)
        print('##############################################################################')
        print(score)


        #Ploting stuff
        tth_sig = []
        ttbb_sherpa = []
        ttbb_powheg = []
        for i,entry in  enumerate(train_label_tmp):
            for j,n in enumerate(entry):
                if n==1:
                    if j==0:
                        ttbb_sherpa.append(y_pred[i][0])
                    elif j==1:
                        ttbb_powheg.append(y_pred[i][0])
                    elif j==2:
                        tth_sig.append(y_pred[i][0])

        plt.clf()

        bin_edges = np.linspace(0, 1, 30)

        plt.hist(tth_sig,     bins=bin_edges, histtype='step', lw=1.5, label='ttH', normed='True', color='#1f77b4')
        plt.hist(ttbb_sherpa, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Sherpa', normed='True', color='#d62728')
        plt.hist(ttbb_powheg, bins=bin_edges, histtype='step', lw=1.5, label='ttbb Powheg', normed='True', color='#008000')
        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(directory_model_properties,'Signal_BKg_plot'+ '.pdf'))

        plt.clf()




        #class_model.evaluate(x=test_data,
        #                     y=test_label_class,
        #                     verbose=1,
        #                     sample_weight=test_weights)
        print('\n')

        # Create 2D plot for the predicted parameter

        # Save cpkt file for this NN
        sess = keras.backend.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, directory_model_properties)

        # Print and save some info about the NN
        network_and_training_properties_string = self._get_network_and_training_properties_string(model_id=model_id,
                                                                                                  number_of_input_neurons=number_of_input_neurons,
                                                                                                  hidden_layers=hidden_layers,
                                                                                                  activation_function_name=activation_function_name,
                                                                                                  dropout_keep_probability=dropout_keep_probability,
                                                                                                  l2_regularization_beta=l2_regularization_beta,
                                                                                                  early_stopping_interval=early_stopping_intervall,
                                                                                                  batch_size_training=batch_size_training,
                                                                                                  early_stopping_epoch=Epoch_end,
                                                                                                  mean_training_time_per_epoch=np.mean(
                                                                                                      times),
                                                                                                  )

        # Save additional Info of NN
        with open(os.path.join(directory_model_properties, 'NN_Info.txt'), 'w') as NN_Info_output_file:
            NN_Info_output_file.write(network_and_training_properties_string)

        print(network_and_training_properties_string, end='')

        with open(os.path.join(directory_model_properties, 'inputVariables.txt'), 'w') as outputfile_input_variables:
            for variable in training_data_set.get_input_variables():
                outputfile_input_variables.write(variable + '\n')

        with open(os.path.join(directory_model_properties, 'preselection.txt'), 'w') as outputfile_preselection:
            outputfile_preselection.write(training_data_set.get_preselection() + '\n')

        print('\n' + '========')
        print('FINISHED')
        print('========' + '\n')

    def _build_model(self,
                     layers,
                     activation_function,
                     number_of_input_neurons,
                     number_of_output_neurons,
                     dropout,
                     l2_regularization,
                     network_type='binary'):
        # Builds the layers to a model

        # Define first layer, which takes all input features
        Inputs= keras.layers.Input(shape=(number_of_input_neurons,))
        X= keras.layers.Activation(activation_function)(X)
        if dropout != 1:
            X = keras.layers.Dropout(dropout)(X)
        for i in range(len(layers) - 1):
            X =keras.layers.Dense(layers[i],
                                activation=activation_function,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization))(X)
            if dropout != 1:
                X= keras.layers.Dropout(dropout)(X)
        # Build last layer, regression only works with 1 parameter
        if network_type == 'binary':
            X= keras.layers.Dense(number_of_output_neurons,
                                         activation='sigmoid',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization))(X)
        else:
            X= keras.layers.Dense(number_of_output_neurons,
                                         activation='softmax',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization))(X)
        return X



    def _get_network_and_training_properties_string(self,
                                                    model_id,
                                                    number_of_input_neurons,
                                                    hidden_layers,
                                                    activation_function_name,
                                                    dropout_keep_probability,
                                                    l2_regularization_beta,
                                                    early_stopping_interval,
                                                    batch_size_training,
                                                    early_stopping_epoch,
                                                    mean_training_time_per_epoch
                                                    ):

        # Prints model properties and returns those properties to be saved in a file
        column_width = 60

        network_and_training_properties = str()

        network_and_training_properties += '{:{width}} {}\n'.format('Model ID:', model_id, width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Network type:', 'regression', width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Number of input variables:',
                                                                    number_of_input_neurons, width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Hidden layers:', hidden_layers, width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Activation function:', activation_function_name,
                                                                    width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Keep probability (dropout):',
                                                                    dropout_keep_probability, width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('L2 regularization:', l2_regularization_beta,
                                                                    width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Early stopping interval:', early_stopping_interval,
                                                                    width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Batch size:', batch_size_training,
                                                                    width=column_width)
        network_and_training_properties += '\n'
        network_and_training_properties += '{:{width}} {}\n'.format('Stopping epoch:', early_stopping_epoch,
                                                                    width=column_width)
        network_and_training_properties += '{:{width}} {:.2f} s\n'.format('Mean training time per epoch:',
                                                                          mean_training_time_per_epoch,
                                                                          width=column_width)
        network_and_training_properties += '\n'

        return network_and_training_properties

    def _plot_history(self, history, path):

        # Plots the history of the loss function
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
                 label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
                 label='Val loss')
        plt.legend()
        plt.savefig(path + "/Plot_history_loss")

        # Save data to file to change plots later
        data_dict = {"Epochs": history.epoch,
                     "Train Loss": list(history.history['mean_absolute_error']),
                     'Val loss': list(history.history['val_mean_absolute_error'])}
        np.savez(path + "/history_data_file.np", data_dict)


from keras.callbacks import Callback


class TimeHistory(keras.callbacks.Callback):
    # Class to track the time spend for each epoch
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print(score)


# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

import os
import datetime

import NNFlow

# ----------------------------------------------------------------------------------------------------


workdir_base = '/storage/9/jschindler/'
name_subdir = 'NN_v1'

number_of_hidden_layers = 3
number_of_neurons_per_layer = 200
hidden_layers = [number_of_neurons_per_layer for i in range(number_of_hidden_layers)]

### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid', 'softplus'
activation_function_name = 'tanh'

early_stopping_intervall = 10

### Parameter for dropout
dropout_keep_probability = 1.

### Parameter for L2 regularization
l2_regularization_beta = 0

batch_size_training = 500
optimizer = None

# ----------------------------------------------------------------------------------------------------


model_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

save_path = os.path.join(workdir_base, name_subdir, 'tth/model_' + model_id)
model_name = name_subdir + '_' + model_id

path_to_training_data_set = '/storage/9/jschindler/data_sets_october_6j2t/training_data_set.hdf'
path_to_validation_data_set = '/storage/9/jschindler/data_sets_october_6j2t/validation_data_set.hdf'
path_to_test_data_set =  '/storage/9/jschindler/data_sets_october_6j2t/test_data_set.hdf'
# ----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
# ----------------------------------------------------------------------------------------------------
train_dict = {'save_path': save_path,
              'model_id': model_id,
              'hidden_layers': hidden_layers,
              'activation_function_name': activation_function_name,
              'dropout_keep_probability': dropout_keep_probability,
              'l2_regularization_beta': l2_regularization_beta,
              'early_stopping_intervall': early_stopping_intervall,
              'path_to_training_data_set': path_to_training_data_set,
              'path_to_validation_data_set': path_to_validation_data_set,
              'path_to_test_data_set': path_to_test_data_set,
              'batch_size_training': batch_size_training,
              }

nEpoch_class=sys.argv[1]
nEpoch_adv = sys.argv[2]
nEpoch_total =sys.argv[3]
t = NeuralNetworkTrainer(int(nEpoch_class),int(nEpoch_adv),int(nEpoch_total))
t.train(**train_dict)


