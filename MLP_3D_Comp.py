
import os, numpy as np, h5py, torch, matplotlib, matplotlib.pyplot as plt
import pyLOM, pyLOM.NN
import math    
import scipy.stats as stats

##Set device
device  = pyLOM.NN.select_device("cuda") 

##Create folder to store outputs

RESUDIR = 'Evaluation_3D_Comp'
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
             
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.95)
        plt.plot(y_true[:, j], 1*y_true[:, j], color='red',alpha=0.4) #R^2=1

        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        #plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=900)


def true_vs_pred_plot_regression(y_true, y_pred, path):
      
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):

        plt.subplot(num_plots, 1, j + 1)

        #obtain m (slope) and b(intercept) of linear regression line
        m, b = np.polyfit(y_true[:, j], y_pred[:, j], 1)
        print('m: ', m)
        print('n: ', b)

        
        plt.plot(y_true[:, j], m*y_true[:, j]+b, color='orange') #regression

        
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.95)

        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        #plt.title(f"Scatterplot for Component {j+1}")
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
    #plt.title("Training Loss vs Epoch")
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


##Load dataset
d = pyLOM.Dataset.load('./KS19/MetalFree_dataset.h5')

T =d.get_variable('T')
nH =d.get_variable('nH')
nhe_nh =d.get_variable('nHe_nH')

X=d['lam'][:,:,0]
Xmin=X.min() #Simulation done at z=0  lam [28512,11,49]

X_sca = dataScaling(X,Xmin,1.01,1e10) #1.01 to avoid zero and 1e10 to avoid machine precision

print('Shape X: ', X.shape)
printAverages('lam',X)

## Obtain pyLOM.NN dataset
input_scaler  = pyLOM.NN.MinMaxScaler()         #MinMaxScalerLog()
output_scaler = pyLOM.NN.MinMaxScaler()         #MinMaxScalerLog()

print('Creating dataset...')

#Create dataset
dataset = pyLOM.NN.Dataset(
    variables_out= (X_sca,),   
    variables_in=np.log(d.xyz), 
    parameters= [d.get_variable('nHe_nH')], 
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=True
)

## Check scaled data

print('Checking scaled data...')


#Scaling for variables in dataset
plt.figure()

plt.subplot(2,4,1)
plt.plot(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[0,:,0],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[0,:,40])
#plt.xlabel(r'$\mathrm{Scaled\  Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.subplot(2,4,2)
plt.plot(dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[0,0,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[0,180,:])
#plt.xlabel(r'$\mathrm{Scaled\  nH \ cm^{-3} }$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')


plt.subplot(2,4,3)
plt.plot(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[3,:,0],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[3,:,40])
#plt.xlabel(r'$\mathrm{Scaled\  Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.subplot(2,4,4)
plt.plot(dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[3,0,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[3,180,:])
#plt.xlabel(r'$\mathrm{Scaled\  nH \ cm^{-3} }$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')


plt.subplot(2,4,5)
plt.plot(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[7,:,0],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[7,:,40])
#plt.xlabel(r'$\mathrm{Scaled\  Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.subplot(2,4,6)
plt.plot(dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[7,0,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[7,180,:])
#plt.xlabel(r'$\mathrm{Scaled\  nH \ cm^{-3} }$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')


plt.subplot(2,4,7)
plt.plot(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[10,:,0],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[10,:,40])
#plt.xlabel(r'$\mathrm{Scaled\  Temperature \ K}$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.subplot(2,4,8)
plt.plot(dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[10,0,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[10,180,:])
#plt.xlabel(r'$\mathrm{Scaled\  nH \ cm^{-3} }$')
#plt.ylabel(r'$\mathrm{Scaled\  |\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')

plt.savefig(RESUDIR + '/Scaled_MetalFree_dataset_variables.png',dpi=900)



#Scaling for dataset
plt.figure()

plt.subplot(2,2,1)
plt.contourf(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[0,:,:],dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[0,:,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[0,:,:],cmap=plt.cm.jet) # [0,:,:] gives the same plots as 2D
cbar = plt.colorbar()

plt.subplot(2,2,2)
plt.contourf(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[3,:,:],dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[3,:,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[3,:,:],cmap=plt.cm.jet) 
cbar = plt.colorbar()

plt.subplot(2,2,3)
plt.contourf(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[7,:,:],dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[7,:,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[7,:,:],cmap=plt.cm.jet) 
cbar = plt.colorbar()

plt.subplot(2,2,4)
plt.contourf(dataset[:][0][:,0].reshape((len(nhe_nh),len(T),len(nH)))[10,:,:],dataset[:][0][:,1].reshape((len(nhe_nh),len(T),len(nH)))[10,:,:],dataset[:][1].reshape((len(nhe_nh),len(T),len(nH)))[10,:,:],cmap=plt.cm.jet) 
cbar = plt.colorbar()

plt.savefig(RESUDIR + '/MLP_MetalFree_Scaled.png',dpi=900)


##Train and test splits

td_train, td_test = dataset.get_splits_by_parameters([0.8, 0.2])  #if parameters=None, dataset.get_splits([0.8, 0.2])

#Training parameters
training_params = {
    "epochs": 750, 
    "lr": 0.0001,
    "lr_gamma": 0.99,
    "lr_scheduler_step": 15,#15,
    "batch_size": 512,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.Adam,
    "print_rate_epoch": 10,
}

sample_input, sample_output = td_train[0]

#Model parameters
model = pyLOM.NN.MLP(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=256,
    n_layers=5,
    p_dropouts=0.15,
)

#Pipeline parameters
pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params=training_params,
)

print('Training...')
training_logs = pipeline.run()


##Save model
pipeline.model.save(os.path.join(RESUDIR,"model_3D_750_eval.pth"))

##Load previous model
#

## Predict and evaluate
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1]])[0]


#Print metrics

evaluator = pyLOM.NN.RegressionEvaluator(tolerance=1e-10)
evaluator(scaled_y, scaled_preds)
evaluator.print_metrics()

true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/3D_Comp_scatter.png')
true_vs_pred_plot_regression(scaled_y, scaled_preds, RESUDIR + '/3D_Comp_regression.png')

plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')

