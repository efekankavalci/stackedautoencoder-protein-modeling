# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
#import seaborn as sns
#import cv2
#import csv
import numpy as np
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"        
import read_coord
import read_angle_long as read_angle
from sklearn import preprocessing
from keras.models import Model
import plot_generator_long as pg


case1_coords=read_coord.read_coord("ca_coords_closed.txt")
case2_coords=read_coord.read_coord("ca_coords_open.txt")
case3_coords=read_coord.read_coord("ca_coords_closedapo.txt")

#case1_coords=case1_coords[0:1913]
#Constructing the standard scaler
alldata=np.array(np.concatenate((case1_coords,case2_coords,case3_coords)))
scaler = preprocessing.StandardScaler().fit(alldata)

#Standardization
case1_coords = scaler.transform(case1_coords)
case2_coords = scaler.transform(case2_coords)
case3_coords = scaler.transform(case3_coords)
            
chain_length=1311
layer_number=2
nodes=[185,90,0,0,8] #Layer neuron numbers, last one is the latent layer 
reg=keras.regularizers.L2(5*1e-4)        
opt = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.85, epsilon=1e-7)
loss_func=loss=keras.losses.MeanSquaredError()

input_features = keras.Input(shape=(chain_length,),name=("input"))
act_func='selu'

for i in range(layer_number):
    if(i==0):
        EHL1=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg, name=str('EHL'+str(i+1)))(input_features)
    elif(i==1):
        EHL2=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg,name=str('EHL'+str(i+1)))(EHL1)
    elif(i==2):
        EHL3=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg,name=str('EHL'+str(i+1)))(EHL2)
    elif(i==3):
        EHL4=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg,name=str('EHL'+str(i+1)))(EHL3)
        
if(layer_number==1):
    LL=keras.layers.Dense(nodes[4], activation=act_func,kernel_regularizer= reg,name='LL')(EHL1)
    EHL1_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL1_y_out"))(EHL1)
elif(layer_number==2):
    LL=keras.layers.Dense(nodes[4], activation=act_func,kernel_regularizer= reg,name='LL')(EHL2)
    EHL1_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL1_y_out"))(EHL1)
    EHL2_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL2_y_out"))(EHL2)
elif(layer_number==3):
    LL=keras.layers.Dense(nodes[4], activation=act_func,kernel_regularizer= reg,name='LL')(EHL3)
    EHL1_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL1_y_out"))(EHL1)
    EHL2_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL2_y_out"))(EHL2)
    EHL3_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL3_y_out"))(EHL3)
elif(layer_number==4):
    LL=keras.layers.Dense(nodes[4], activation=act_func,kernel_regularizer= reg,name='LL')(EHL4)
    EHL1_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL1_y_out"))(EHL1)
    EHL2_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL2_y_out"))(EHL2)
    EHL3_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL3_y_out"))(EHL3)
    EHL4_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("EHL4_y_out"))(EHL4)


LL_y_output=keras.layers.Dense(1, activation='sigmoid',kernel_regularizer= reg,name=("LL_y_out"))(LL)


for i in reversed(range(layer_number)):
    if(i==layer_number-1):
        DHL=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg,name=str('DHL'+str(i+1)))(LL)
    else:
        DHL=keras.layers.Dense(nodes[i], activation=act_func, kernel_regularizer= reg,name=str('DHL'+str(i+1)))(DHL)

decoded = keras.layers.Dense(1311, activation=act_func, kernel_regularizer= reg,name='decoded')(DHL)

# Reading angles from closed data and open data, and scaling
closed_train_angles=read_angle.read_angle('closed_angle.txt')
#closed_train_angles=closed_train_angles[0:1913]
open_train_angles=read_angle.read_angle('open_angle.txt')
y_train_angles=np.concatenate((closed_train_angles,open_train_angles))   

angle_scaler = preprocessing.MinMaxScaler()
angle_scaler.fit(y_train_angles)
y_train=angle_scaler.transform(y_train_angles)

test_angles=read_angle.read_angle('apo_angle.txt')
y_test=angle_scaler.transform(test_angles)

x_train=np.array(np.concatenate((case1_coords,case2_coords)))
x_test=np.array(case3_coords)

"""
def y_loss_fn(y_true, y_pred):
    constant=1
    squared_difference = constant*tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference)
{"layer #": 3, "layer1 node": 225, "layer2 node": 60, "layer3 node": 15, "layer4 node": 0, "latent-layer node": 5, "activation": "selu", "learning rate": 0.001, "momentum": 0.85, "regularization": 0.0001, "epoch": 500, "minibatch": 512}
"""
def y_loss_fn(y_true, y_pred):
    constant=0.5
    return constant * loss_func(y_true, y_pred)

if(layer_number==1):
    output_array=[decoded,LL_y_output,EHL1_y_output]
    fit_array=[x_train,y_train,y_train]
    loss_array=[loss_func,y_loss_fn,y_loss_fn]
elif(layer_number==2):
    output_array=[decoded,LL_y_output,EHL1_y_output,EHL2_y_output]
    fit_array=[x_train,y_train,y_train,y_train]
    loss_array=[loss_func,y_loss_fn,y_loss_fn,y_loss_fn]
elif(layer_number==3):
    output_array=[decoded,LL_y_output,EHL1_y_output,EHL2_y_output,EHL3_y_output]
    fit_array=[x_train,y_train,y_train,y_train,y_train]
    loss_array=[loss_func,y_loss_fn,y_loss_fn,y_loss_fn,y_loss_fn]
elif(layer_number==4):
    output_array=[decoded,LL_y_output,EHL1_y_output,EHL2_y_output,EHL3_y_output,EHL4_y_output]
    fit_array=[x_train,y_train,y_train,y_train,y_train,y_train]
    loss_array=[loss_func,y_loss_fn,y_loss_fn,y_loss_fn,y_loss_fn,y_loss_fn]



autoencoder = keras.Model(inputs=input_features, outputs = output_array)
autoencoder.compile(optimizer=opt, loss=loss_array)

autoencoder.fit(x_train, fit_array,
                epochs=100,
                batch_size=16,
                shuffle=True)
                #validation_data=(x_train, fit_array))
#history.history
#autoencoder.summary()

LL_model = Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('LL').output)
LL_pred = LL_model.predict(scaler.inverse_transform(x_train))
LL_pred_test = LL_model.predict(scaler.inverse_transform(x_test))


descaled_predictions= np.array(scaler.inverse_transform(autoencoder.predict(x_test)[0]))
descaled_x_test=np.array(scaler.inverse_transform(x_test))

descaled_train_predictions= np.array(scaler.inverse_transform(autoencoder.predict(x_train)[0]))
descaled_x_train=np.array(scaler.inverse_transform(x_train))

y1_pred_test= np.array((autoencoder.predict(x_test)[1]))
y2_pred_test= np.array((autoencoder.predict(x_test)[2]))
y3_pred_test= np.array((autoencoder.predict(x_test)[3]))


y1_pred_train= np.array((autoencoder.predict(x_train)[1]))
y2_pred_train= np.array((autoencoder.predict(x_train)[2]))
y3_pred_train= np.array((autoencoder.predict(x_train)[3]))
    

def rmse_y(y_predict,y_test):
    return np.sqrt(np.mean(np.square(y_predict-y_test)))

def plot_rmse_y():
    fig, ax = plt.subplots()
    xs=[]
    ys=[]
    for i in range(layer_number+1):
        label_ax=str("RMSE of"+str(i+1)+"y node")
        y_rmse=np.array((autoencoder.predict(x_test)[i+1]))
        xs=np.append(xs,i+1)
        ys=np.append(ys,rmse_y(y_rmse,y_test))
    ax.scatter(xs,ys,s=40)
    #ax.xticks(np.arange(min(xs), max(xs)+1, 1.0))
    ax.set_xlabel('Y neuron outputs',fontsize='medium')
    ax.set_ylabel('RMSE',fontsize='medium')
    ax.grid()
    plt.show()

plot_rmse_y()

        
def plot_test():    
    fig, ax = plt.subplots()
    plt.plot(y_test,label='test')
    plt.plot(y1_pred_test,label='predict1')
    #plt.plot(LL_y_model.predict(x_test),label='LL y')
    plt.plot(y2_pred_test,label='predict2')
    #plt.plot(y3_pred_test,label='predict3')
    #plt.plot(EHL2_y_model.predict(x_test),label='EHL2 y')
    plt.xlabel('Snapshots in test data ')
    plt.ylabel('MinMax Scaled Angle ')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    ax.grid()
    plt.show()

def plot_test_average():
    fig, ax = plt.subplots()
    plt.plot(y_test,label='Test dataset')

    plt.plot((y1_pred_test+y2_pred_test)/2,label='prediction of 2 y output neuron average')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    plt.xlabel('Snapshots in test data ')
    plt.ylabel('MinMax Scaled Angle ')
    plt.title('Test angle vs. average of predicted angles at y neuron 1&2')
    ax.grid()
    plt.show()

    
def plot_train():
    fig, ax = plt.subplots()
    plt.plot(y_train,label='train')
    plt.plot(y1_pred_train,label='predict1')
    #plt.plot(LL_y_model.predict(x_test),label='LL y')
    plt.plot(y2_pred_train,label='predict2')
    #plt.plot(y3_pred_train,label='predict3')
    plt.xlabel('Snapshots in train data ')
    plt.ylabel('MinMax Scaled Angle ')
    #plt.plot(EHL2_y_model.predict(x_test),label='EHL2 y')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    ax.grid()
    plt.show()
    
def plot_bottleneck(LL_pred):
    fig, ax = plt.subplots()
    for i in range(0, nodes[4]):
        #ll_scaler = preprocessing.StandardScaler().fit(LL_pred[:,i].reshape(-1,1))
        #latent_node=ll_scaler.transform(LL_pred[:,i].reshape(-1,1))
        latent_node=LL_pred[:,i]
        plt.plot(latent_node,label='latent-param'+str(i+1),linewidth=0.7)
    plt.xlabel('Snapshot in dataset')
    plt.ylabel('Bottleneck neuron value')
    #plt.plot(EHL2_y_model.predict(x_test),label='EHL2 y')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    ax.grid()
    plt.show()

plot_train()
plot_test()
plot_bottleneck(LL_pred)
plot_bottleneck(LL_pred_test)

#plot_test_average()


def residue_distance(descaled_predictions, descaled_x_test):
    residue_hypotenus=np.zeros((len(descaled_x_test),int(len(descaled_x_test[0])/3)))
    diff=descaled_predictions-descaled_x_test
    for i in range(len(x_test)):
        for j in range(int(len(descaled_x_test[0])/3)):
            residue_hypotenus[i][j]=np.sqrt(np.sum(np.square(diff[i][(j*3):((j+1)*3)])))
    return(residue_hypotenus)      
        
residue_hypotenus=residue_distance(descaled_predictions, descaled_x_test)
print('Average hypotenus length of all residues in all snapshots is:%.2f' % (np.mean(residue_hypotenus)))

def residue_sqrd_distance(descaled_predictions, descaled_x_test):
    residue_se=np.zeros((len(descaled_x_test),int(len(descaled_x_test[0])/3)))
    diff=descaled_predictions-descaled_x_test
    for i in range(len(descaled_x_test)):
        for j in range(int(len(x_test[0])/3)):
            residue_se[i][j]=(np.sum(np.square(diff[i][(j*3):((j+1)*3)])))
    return(residue_se)
residue_se = residue_sqrd_distance(descaled_predictions, descaled_x_test)
print('RMSE of all residues in all snapshots in test data is:%.2f' % np.sqrt((np.mean(residue_se))))
residue_se_train =residue_sqrd_distance(descaled_train_predictions, descaled_x_train)
print('RMSE of all residues in all snapshots in trained data is:%.2f' % np.sqrt((np.mean(residue_se_train))))

pg.residue_plots(residue_se)

#------------------OLD NOTES-------------------------------------------------
#rmse_snapshots=np.mean(rmse_residues,axis=1)
#Getting weights for a layer
#weights  = autoencoder.get_layer('layer_1').get_weights()

"""
# Custom Network Structure Outputs 

final_layer_LL_y=(autoencoder.predict(x_train))[1]
final_layer_EHL2_y=(autoencoder.predict(x_train))[2]

# For observing output in extra neurons with sigmoid 
#intermediate_output = intermediate_layer_model.predict([x_test[0].reshape((-1,1311,))])
intermediate_output = intermediate_layer_model.predict(x_train)
#ys=intermediate_output[:,-1]
"""

"""
#errors=np.square(np.subtract(predictions,descaled_test))
#error_rows=np.mean(errors,axis=1)

"""
    
