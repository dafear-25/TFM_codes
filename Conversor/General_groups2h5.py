

#!/usr/bin/env python
#
# Conversor of cooling tables data file to HDF5 file. Case General for xi, solar, and total metals groups
#
#Last version 30/07/2025

import numpy as np, h5py
import glob
import ast
import pyLOM

## Functions 


def generate_dataset_groups(path_tables,group,file_out):
   
   data=glob.glob(path_tables)
   data.sort()
   str_d = str(data) 
   testarray = ast.literal_eval(str_d)

    #Headers name on each element
   group_name = group  #Carbon, ..., Iron

   nH = "Hydrogen_density_bins"  
   T = "Temperature_bins"
   

   Species="Species_names"
   NetCooling = "Net_Cooling"

    #General values in all tables
   header_name = "Header"
   z = "Redshift"
   n_den_bins = "Number_of_density_bins"
   n_temp_bins = "Number_of_temperature_bins"

   #Get array sizes
   len_nh = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_den_bins}"]).item() # 81
   len_T = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_temp_bins}"]).item() #352
   
   len_z = len(testarray) #49

  #Create arrays

  #Array for z
   dataset_z = np.ones((len_z,1))
   
   
  #Constant arrays are created using the data in z=0
   dataset_nh = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{nH}"])
   dataset_T = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{T}"])
   
  #Array for cooling values
   dataset_Cool = np.zeros((len_z,len_T,len_nh))

   for i in range(0,len_z): 
     #For each table it is stored z and its cooling values

     dataset_z[i] = np.array(h5py.File(testarray[i], "r")[f"{header_name}/{z}"]) 
       
     dataset_Cool[i,:,:] = np.array(h5py.File(testarray[i], "r")[f"{group_name}/{NetCooling}"])

     #Close each file once saved
     h5py.File(testarray[i], "r").close

   
   #Correct dimensions 
   dataset_z=np.squeeze(dataset_z,axis=1)
   print('z: ', dataset_z)



   # Build xyz coordinated from T and nH
   xx,yy = np.meshgrid(dataset_T,dataset_nh,indexing='ij')
   xyz = np.zeros((len(dataset_T)*len(dataset_nh),2),dataset_T.dtype) #Agafar variable len corresponent
   xyz[:,0] = xx.flatten()
   xyz[:,1] = yy.flatten()

   print('Shape xyz: ',xyz.shape) #352*81,2 - pairs T-nH
  

   #The points on the arrays must be ordered consecutively 
   pointO=np.arange(xyz.shape[0])
   print('pointO:', pointO.shape) #(28512,)


   
   #The number of points matches the first dimension of xyz and lam
   npoints=xyz.shape[0]

   #Table with only one partition with the same number of points for elements
   ptable=pyLOM.PartitionTable.new(1,npoints,npoints)
   print(ptable)


   #Create matrix with all the cooling values
   lam=np.zeros((len_T*len_nh,len_z))

  #The groups generated for the diferent z are joined 
   #lam must be (28512,49) and Cool_set is (49,352,81)

   for i in range(0,len_z):
    
      lam[:,i]=dataset_Cool[i,:,:].flatten(order='C')
      

   #print(lam)
   print(lam.shape)


   ##Creation pyLOM dataset

   d=pyLOM.Dataset(xyz=xyz,ptable=ptable,order=pointO,point=True,
    #Varibles associated to the dimensions of the matrix lam
    vars={
        #The first dimension of lam is associated with idim=0
        
        'T' : {'idim':0, 'value':dataset_T},

        'nH' : {'idim':0, 'value':dataset_nh},


        'z' : {'idim':1, 'value':dataset_z},
        },

        lam = {'ndim':1, 'value':lam}   #Camp escalar és 1 perquè és 1 
    )

   print(d)


    ##Store dataset
   d.save(file_out,mode='w') #w to overwrite previous


   
## Parameters 
INFILE = "./KS19/z_[0-9]????.hdf5"
OUTFILE = "./KS19/Solar_dataset.h5" 
group_name= "Solar"


generate_dataset_groups(INFILE, group_name, OUTFILE)

pyLOM.cr_info()
