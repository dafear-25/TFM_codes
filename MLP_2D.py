
#Model 2D -MetalFree


#Importar dataset i folder

import os, numpy as np, h5py, torch, matplotlib, matplotlib.pyplot as plt
import pyLOM, pyLOM.NN
import math    

##Set device
device  = pyLOM.NN.select_device("cuda") 

##Create folder to store outputs

RESUDIR = 'Model_2D_MetalFree'
pyLOM.NN.create_results_folder(RESUDIR)


##Functions

#Function to plot some statistics (min,max,mean) of the dataset
def printAverages(str,scaf):
    
    pyLOM.pprint(0,str + ' min=%.4e, max=%.4e, avg=%.4e'%(np.min(scaf),np.max(scaf),np.mean(scaf)),scaf.dtype,scaf.shape,flush=True)


#Auxiliary function to plot the true vs predicted values
def true_vs_pred_plot(y_true, y_pred, path):
      
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=900)



#Auxiliary function to plot the training and test loss
def plot_train_test_loss(train_loss, test_loss, path):
       
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path,dpi=900)


#Function to scale dataset
def dataScaling(scaf,offset,f1,f2):
    return np.log10((scaf + f1*abs(offset))*f2)

#Function to reverse scaling dataset
def dataInverseScaling(scaf,offset,f1,f2):
    return np.power(10,scaf)/(f2) -f1*np.abs(offset)



d = pyLOM.Dataset.load('./KS19/MetalFree_dataset.h5')


T =d.get_variable('T')
nH =d.get_variable('nH')
nhe_nh =d.get_variable('nHe_nH')

X=d['lam'][:,0,0]#.reshape((len(T),len(nH)))
Xmin=X.min() #Simulation done at z=0  lam [28512,11,49]

X_sca = dataScaling(X,Xmin,1.01,1e10) #1.01 to avoid zero and 1e10 to avoid machine precision

print('Shape X: ', X.shape)
printAverages('lam',X)

## Obtain pyLOM.NN dataset
input_scaler  = pyLOM.NN.MinMaxScaler()         
output_scaler = pyLOM.NN.MinMaxScaler()         

print('Creating dataset...')

#Create dataset
dataset = pyLOM.NN.Dataset(
    variables_out= (X_sca,),   
    variables_in=np.log(d.xyz), 
    parameters= None, 
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=True
)

# Check scaled data
plt.figure(figsize=[2*6.4, 4.8])

plt.subplot(1,2,1)
plt.plot(dataset[:][0][:,0].reshape((len(T),len(nH)))[:,0],dataset[:][1].reshape((len(T),len(nH)))[:,40])
plt.xlabel(r'$\mathrm{Scaled\  Temperature }$')
plt.ylabel(r'$\mathrm{Scaled\  \Lambda}$')

plt.subplot(1,2,2)
plt.plot(dataset[:][0][:,1].reshape((len(T),len(nH)))[0,:],dataset[:][1].reshape((len(T),len(nH)))[180,:])
plt.xlabel(r'$\mathrm{Scaled\  nH }$')
plt.ylabel(r'$\mathrm{Scaled\  \Lambda}$')

plt.savefig(RESUDIR + '/Scaled_MetalFree_dataset_variables.png',dpi=900)

plt.figure()
plt.contourf(dataset[:][0][:,0].reshape((len(T),len(nH))),dataset[:][0][:,1].reshape((len(T),len(nH))),dataset[:][1].reshape((len(T),len(nH))),cmap=plt.cm.jet)
cbar = plt.colorbar()
plt.savefig(RESUDIR + '/MLP_MetalFree_Scaled.png',dpi=900)

td_train, td_test = dataset.get_splits([0.8, 0.2])

training_params = {
    "epochs": 1500,
    "lr": 0.0001,
    "lr_gamma": 0.99,
    "lr_scheduler_step": 15,
    "batch_size": 512,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.Adam,
    "print_rate_epoch": 10,
}

sample_input, sample_output = td_train[0]
model = pyLOM.NN.MLP(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=256,
    n_layers=5,
    p_dropouts=0.15,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params=training_params,
)

training_logs = pipeline.run()

##Save model
pipeline.model.save(os.path.join(RESUDIR,"model_2D_1500_MetalFree.pth"))

##Load previous model



## Predict and evaluate
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1]])[0]


#Print metrics

evaluator = pyLOM.NN.RegressionEvaluator(tolerance=1e-10)
evaluator(scaled_y, scaled_preds)
evaluator.print_metrics()


true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')


#

plt.figure(figsize=[2*6.4, 4.8])

plt.subplot(1,2,1)
plt.plot(td_test[:][0][:,0],td_test[:][1],'.',label='data')
plt.plot(td_test[:][0][:,0],preds,'.',label='MLP')
plt.xlabel(r'$\mathrm{Scaled\  T }$')
plt.ylabel(r'$\mathrm{Scaled\  \Lambda}$')

plt.subplot(1,2,2)
plt.plot(td_test[:][0][:,1],td_test[:][1],'.',label='data')
plt.plot(td_test[:][0][:,1],preds,'.',label='MLP')
plt.xlabel(r'$\mathrm{Scaled\  nH }$')
plt.ylabel(r'$\mathrm{Scaled\  \Lambda}$')
plt.legend()

plt.savefig('MLP_MetalFree_traintest.png',dpi=900)




print('Evaluating model full dataset...')

## Evaluate the model with the full dataset
preds = model.predict(dataset, batch_size=250)
scaled_x     = input_scaler.inverse_transform(dataset[:][0])
scaled_y     = output_scaler.inverse_transform(dataset[:][1])
scaled_preds = output_scaler.inverse_transform(preds)


#Evaluate variable 1D scaled
plt.figure()

plt.plot(scaled_x[:,0].reshape((len(T),len(nH)))[:,0],scaled_y.reshape((len(T),len(nH)))[:,40])
plt.plot(scaled_x[:,0].reshape((len(T),len(nH)))[:,0],scaled_preds.reshape((len(T),len(nH)))[:,40])

plt.savefig(RESUDIR + '/MLP_MetalFree_scaled_nH_vs_scaled_y_preds.png',dpi=900)



#Evaluate variables scaled contourplot
plt.figure()

plt.subplot(1,2,1)
plt.contourf(scaled_x[:,0].reshape((len(T),len(nH))),scaled_x[:,1].reshape((len(T),len(nH))),scaled_y.reshape((len(T),len(nH))),cmap=plt.cm.jet) #watch transposed scaled_y
plt.subplot(1,2,2)
plt.contourf(scaled_x[:,0].reshape((len(T),len(nH))),scaled_x[:,1].reshape((len(T),len(nH))),scaled_preds.reshape((len(T),len(nH))),cmap=plt.cm.jet) #watch transposed scaled_preds

plt.savefig(RESUDIR + '/MLP_MetalFree_scaled_T_nH_vs_scaled_y_preds_contour.png',dpi=900)


#Reverse varibles and data scaling
scaled_x     = np.exp(scaled_x)


scaled_y= dataInverseScaling(scaled_y,Xmin,1.01,1e10)

scaled_preds=dataInverseScaling(scaled_preds,Xmin,1.01,1e10)

printAverages('X',X)
printAverages('scaled_preds',scaled_preds)


#Evaluate variable 1D with output
plt.figure()

plt.loglog(scaled_x[:,0].reshape((len(T),len(nH)))[:,0],np.abs(scaled_y.reshape((len(T),len(nH)))[:,40]), label='data')
plt.loglog(scaled_x[:,0].reshape((len(T),len(nH)))[:,0],np.abs(scaled_preds.reshape((len(T),len(nH)))[:,40]), label='MLP')
plt.title('Normalized, absolute, net cooling rates - Metal Free')
plt.xlabel(r'$\mathrm{Temperature \ K}$')
plt.ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/MLP_MetalFree_T_vs_lam_vs_preds.png',dpi=900)




#Evaluate dataset vs predictions contourplot
plt.figure()

plt.subplot(1,2,1)
plt.contourf(scaled_x[:,0].reshape((len(T),len(nH))),scaled_x[:,1].reshape((len(T),len(nH))),np.abs(scaled_y.reshape((len(T),len(nH)))),locator=matplotlib.ticker.LogLocator(),cmap=plt.cm.jet)
plt.xscale('log')
plt.yscale('log')
#plt.xlabel(r'$\mathrm{Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Density \ kg/m^3}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.subplot(1,2,2)
plt.contourf(scaled_x[:,0].reshape((len(T),len(nH))),scaled_x[:,1].reshape((len(T),len(nH))),np.abs(scaled_preds.reshape((len(T),len(nH)))),locator=matplotlib.ticker.LogLocator(),cmap=plt.cm.jet)
plt.xscale('log')
plt.yscale('log')
#plt.xlabel(r'$\mathrm{Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Density \ kg/m^3}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/MLP_MetalFree_scaled_x_vs_scaled_y_and_scaled_preds.png',dpi=900)



## Generate more points on the airfoil and evaluate
# Interpolate points
T_new  = np.interp(np.arange(500),np.arange(len(T)),T)
nH_new = np.interp(np.arange(200),np.arange(len(nH)),nH)

# Build xyz from T and nH
xx,yy = np.meshgrid(T_new,nH_new,indexing='ij')
xyz = np.zeros((len(T_new)*len(nH_new),2),T.dtype)
xyz[:,0] = xx.flatten()
xyz[:,1] = yy.flatten()

# Generate new dataset
dataset_2 = pyLOM.NN.Dataset(
    variables_out=(np.zeros((len(T_new),len(nH_new))),),
    variables_in=np.log(xyz),
    parameters=None,
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=True
)


print('Evaluating model more points...')

# Evaluate new dataset with saved model
preds_2 = model.predict(dataset_2, batch_size=250)

scaled_x2    = input_scaler.inverse_transform(dataset_2[:][0])
scaled_preds_2 = output_scaler.inverse_transform(preds_2)


plt.figure()

plt.contourf(scaled_x2[:,0].reshape((len(T_new),len(nH_new))),scaled_x2[:,1].reshape((len(T_new),len(nH_new))),scaled_preds_2.reshape((len(T_new),len(nH_new))),cmap=plt.cm.jet)

plt.savefig(RESUDIR + '/MLP_MetalFree_scaled_x2_vs_scaled_preds.png',dpi=900)


#Reverse varibles and data scaling


scaled_preds_2=dataInverseScaling(scaled_preds_2,Xmin,1.01,1)

printAverages('X',X)
printAverages('scaled_preds_2',scaled_preds_2)


plt.figure() #

#plt.subplot(2,2,1)
plt.contourf(T_new,nH_new,(np.abs(scaled_preds_2.reshape((len(T_new),len(nH_new))))).T,locator=matplotlib.ticker.LogLocator(),cmap=plt.cm.jet)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{nH \ (cm^{-3})}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/MLP_MetalFree_more_points_out.png',dpi=900)



print('Error calculation...')

error_0= ( (( (scaled_preds.reshape((len(T),len(nH))))  -X.reshape((len(T),len(nH))) )/(X.reshape((len(T),len(nH)))))*100 )


print('Final plots...')

#Mesh plot original data, predicted data and associated error

plt.figure()

plt.contourf(T,nH,(np.abs(X.reshape((len(T),len(nH))))).T,locator=matplotlib.ticker.LogLocator(),cmap=plt.cm.jet)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{nH \ (cm^{-3})}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/MLP_MetalFree_original_data.png',dpi=900)


#Plot predicted data with original points
plt.figure()
#plt.subplot(2,2,1)
plt.contourf(T,nH,(np.abs(scaled_preds.reshape((len(T),len(nH))))).T,locator=matplotlib.ticker.LogLocator(),cmap=plt.cm.jet)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{nH \ (cm^{-3})}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/MLP_MetalFree_out_predicted.png',dpi=900)



#Plot error between predicted and original data
plt.figure()

plt.contourf(T,nH,(error_0).T,cmap='RdYlGn')#plt.colormaps.diverging)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{nH \ (cm^{-3})}$')
plt.xlim([1e2, 1e9])
plt.ylim([1e-8, 1e0])
cbar = plt.colorbar()

#cbar.ax.set_ylabel(r'Absolute difference between original and predicted data')

plt.savefig(RESUDIR + '/MLP_MetalFree_out_error.png',dpi=900)

pyLOM.cr_info()
plt.show()
