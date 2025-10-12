
#!/usr/bin/env python
#
# Model for one variable. Temperature (T). Case Metal Free
#
# Last version 06/06/2025


import os, numpy as np, h5py, torch, matplotlib, matplotlib.pyplot as plt
import pyLOM, pyLOM.NN
import math    
import scipy.stats as stats

##Set device
device  = pyLOM.NN.select_device("cuda") 

##Create folder to store outputs

RESUDIR = 'Model_1D_T'
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
        #obtain m(slope) and b(intercept) of linear regression line
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


d = pyLOM.Dataset.load('./KS19/MetalFree_dataset.h5')

T =d.get_variable('T')
nH =d.get_variable('nH')
nhe_nh =d.get_variable('nHe_nH')

X=d['lam'][:,0,0].reshape((len(T),len(nH))) #Simulation done at z=0  lam [28512,11,49]
Xmin=X.min() 

X_sca = dataScaling(X,Xmin,1.01,1e10) #1.01 to avoid zero and 1e10 to avoid machine precision

print('Shape X: ', X.shape)
printAverages('lam',X)

## Obtain pyLOM.NN dataset
input_scaler  = pyLOM.NN.MinMaxScaler()         
output_scaler = pyLOM.NN.MinMaxScaler()         

print('Creating dataset...')

#Create dataset
dataset = pyLOM.NN.Dataset(
    variables_out= (X_sca[:,40],),  
    variables_in=np.expand_dims(np.log(T),axis=1), 
    parameters= None, 
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=True
)


# Check scaled data
plt.figure()
plt.plot(dataset[:][0][:,0],dataset[:][1])
plt.savefig(RESUDIR + '/scaled_data.png', dpi=900)

td_train, td_test = dataset.get_splits([0.7, 0.3])

training_params = {
    "epochs": 1500,
    "lr": 0.001,
    "lr_gamma": 0.98,
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
    hidden_size=512,
    n_layers=6,
    p_dropouts=0.05,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params=training_params,
)

training_logs = pipeline.run()

##Save model
pipeline.model.save(os.path.join(RESUDIR,"model_1D_1500_T.pth"))

##Load previous model
#

## Predict and evaluate
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1]])[0]

evaluator = pyLOM.NN.RegressionEvaluator(tolerance=1e-10)
evaluator(scaled_y, scaled_preds)
evaluator.print_metrics()


true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/1D_T_scatter.png')
true_vs_pred_plot_regression(scaled_y, scaled_preds, RESUDIR + '/1D_T_regression.png')

plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/1D_T_train_logs.png')

plt.figure()
plt.plot(td_test[:][0][:,0],td_test[:][1],'.')
plt.plot(td_test[:][0][:,0],preds,'.')
plt.savefig(RESUDIR + '/dataset_vs_predicted.png', dpi=900)


## Evaluate the model with the full dataset
preds = model.predict(dataset, batch_size=250)
scaled_x     = input_scaler.inverse_transform([dataset[:][0]])[0]
scaled_y     = output_scaler.inverse_transform([dataset[:][1]])[0]
scaled_preds = output_scaler.inverse_transform([preds])[0]

plt.figure()
plt.plot(scaled_x,scaled_y)
plt.plot(scaled_x,scaled_preds)
plt.savefig(RESUDIR + '/full_dataset_vs_predicted.png', dpi=900)


#Reverse varibles and data scaling
scaled_x     = np.exp(scaled_x)
scaled_y= dataInverseScaling(scaled_y,Xmin,1.01,1e10)
scaled_preds=dataInverseScaling(scaled_preds,Xmin,1.01,1e10)

printAverages('X',X[:,40])
printAverages('scaled_preds',scaled_preds)


plt.figure()
plt.loglog(scaled_x,np.abs(scaled_y))
plt.loglog(scaled_x,np.abs(scaled_preds))
plt.title('Normalized, absolute, net cooling rates - Metal Free')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')
plt.savefig(RESUDIR + '/model_dataset.png', dpi=900)


## Generate more points on the cooling function and evaluate
# Interpolate points
T_new = np.interp(np.arange(500),np.arange(len(T)),T)

# Generate a dataset
dataset_2 = pyLOM.NN.Dataset(
    variables_out=(np.zeros((len(T_new),)),),
    variables_in=np.expand_dims(np.log(T_new),axis=1),
    parameters=None,
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=True
)

# Evaluate
preds_2 = model.predict(dataset_2, batch_size=250)
scaled_x2    = input_scaler.inverse_transform([dataset_2[:][0]])[0]
scaled_preds_2 = output_scaler.inverse_transform([preds_2])[0]

plt.figure()
plt.plot(scaled_x2,scaled_preds_2)
plt.savefig(RESUDIR + '/scaled_newdata.png', dpi=900)

scaled_preds_2 = dataInverseScaling(scaled_preds_2,Xmin,1.01,1e10)

plt.figure()
#plt.loglog(T,np.abs(X[0,:,40]),label="Original data")
#plt.loglog(T,np.abs(X[:,0].reshape((len(T),len(nH)))[:,40]),label="Original data")
plt.loglog(T,np.abs(X[:,40]),label="Original data")
plt.loglog(T,np.abs(scaled_preds),label="MLP prediction")
#plt.title('Normalized, absolute, net cooling rates - Metal Free with prediction model and original data')
plt.xlabel(r'$\mathrm{T \ (K)}$')
plt.ylabel(r'$\mathrm{|\Lambda \ / \ n_{H}^2| \ (erg\ cm^3\ s^{-1})}$')
plt.legend()
plt.savefig(RESUDIR + '/1D_T_real_predicted.png', dpi=900)

pyLOM.cr_info()
plt.show()
