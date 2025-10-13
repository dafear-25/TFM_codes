
#!/usr/bin/env python
#
# Conversor of cooling tables data file to HDF5 file. Case Metal Free
#
# Last version 16/04/2025

import numpy as np, h5py
import glob
import ast
import pyLOM



def generate_dataset_MetalFree(path_tables,file_out):
   
  data=glob.glob(path_tables)
  data.sort()
  str_d = str(data) 
  testarray = ast.literal_eval(str_d)

  #Headers name on each element
  group_name = "Metal_free"  

  nH = "Hydrogen_density_bins"  
  T = "Temperature_bins"
  nhe_nh ="Helium_number_ratio_bins"

  NetCooling = "Net_Cooling"

  #General values in all tables
  header_name = "Header"
  z = "Redshift"
  n_den_bins = "Number_of_density_bins"
  n_temp_bins = "Number_of_temperature_bins"
  n_nHenH = "Number_of_helium_fractions"

  #Get array sizes
  len_nh = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_den_bins}"]).item() # .item converts from scalar array to scalar index
  len_T = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_temp_bins}"]).item() #352
  len_nHenH = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_nHenH}"]).item() #11
  len_z = len(testarray) #49

  print(len_T,len_nh,len_nHenH,len_z)

  #Create arrays

  #Array for z
  dataset_z = np.ones((len_z,1))

  #Constant arrays are created using the data in z=0
  dataset_nh = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{nH}"])
  dataset_T = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{T}"])
  dataset_nhe_nh = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{nhe_nh}"])

  print('Shape nHe/nH: ', dataset_nhe_nh.shape)

  #Array for cooling values
  dataset_Cool = np.zeros((len_z,len_nHenH,len_T,len_nh))

  for i in range(0,len_z):
     
    #For each table it is stored z and its cooling values

    dataset_z[i] = np.array(h5py.File(testarray[i], "r")[f"{header_name}/{z}"]) 
       
    dataset_Cool[i,:,:,:] = np.array(h5py.File(testarray[i], "r")[f"{group_name}/{NetCooling}"])

    #Close each file once saved
    h5py.File(testarray[i], "r").close


  print('Shape z: ', dataset_z.shape)

  #Correct dimensions
  dataset_z=np.squeeze(dataset_z,axis=1)
  print('Shape z: ', dataset_z.shape) #(49,)


  # Build xyz coordinated from T and nH
  xx,yy = np.meshgrid(dataset_T,dataset_nh,indexing='ij')
  xyz = np.zeros((len(dataset_T)*len(dataset_nh),2),dataset_T.dtype) 
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
  lam=np.zeros((len_T*len_nh,len_nHenH,len_z))

  #The groups generated for the diferent z are joined 
  #lam must be (28512,11,49) and Cool_set (49,11,352,81)

  for i in range(0,len_z):
     for j in range(0,len_nHenH):

      lam[:,j,i]=dataset_Cool[i,j,:,:].flatten(order='C')
        

  print(lam)
  print(lam.shape)

  ##Creation pyLOM dataset

  d=pyLOM.Dataset(xyz=xyz,ptable=ptable,order=pointO,point=True,
    #Variables associated to the dimensions of the matrix lam
    vars={
      #The first dimension of lam is associated with idim=0
      
      
      'T' : {'idim':0, 'value':dataset_T},

      'nH' : {'idim':0, 'value':dataset_nh},


      'nHe_nH' : {'idim':0, 'value':dataset_nhe_nh},

      'z' : {'idim':1, 'value':dataset_z},
        },

      lam = {'ndim':1, 'value':lam}   #Scalar field 
    )

  print(d)

  ##Store dataset
  d.save(file_out,mode='w') #w to overwrite previous




"""

## Parameters 
OUTFILE = "./MetalFree_dataset_TnH.h5" 

#MN5: /home/bsc/bsc320267/scratch/datasets/New_datasets
#Local:./MetalFree_dataset.h5


## Load data from Cooling tables

#Read cooling tables

d = glob.glob("/home/bsc/bsc320267/scratch/Cooling_tables/HM12/z_[0-9]????.hdf5") #Change name of the variable here

#MN5: /home/bsc/bsc320267/scratch/Cooling_tables/HM12/
#Local: /home/dferrerarg/data_proves/dades/taules_noves

#Order the paths
d.sort()

## Files management
str_d = str(d) #Conversion to string
#print(type(str_d))

testarray = ast.literal_eval(str_d)

#Headers name on each element
group_name = "Metal_free"  

nH = "Hydrogen_density_bins"  
T = "Temperature_bins"
nhe_nh ="Helium_number_ratio_bins"

# species="Species_names"
NetCooling = "Net_Cooling"

#General values in all tables
header_name = "Header"
z = "Redshift"
n_den_bins = "Number_of_density_bins"
n_temp_bins = "Number_of_temperature_bins"
n_nHenH = "Number_of_helium_fractions"

#Get array sizes
len_nh = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_den_bins}"]).item() # .item converts from scalar array to scalar index
len_T = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_temp_bins}"]).item() #352
len_nHenH = np.array(h5py.File(testarray[0], "r")[f"{header_name}/{n_nHenH}"]).item() #11
len_z = len(testarray) #49

print(len_T,len_nh,len_nHenH,len_z)

#Create arrays
#Array for z
dataset_z = np.ones((len_z,1))

#Constant arrays are created using the data in z=0
dataset_nh = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{nH}"])
dataset_T = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{T}"])
dataset_nhe_nh = np.array(h5py.File(testarray[0], "r")[f"{group_name}/{nhe_nh}"])

print('Shape nHe/nH: ', dataset_nhe_nh.shape)


#Array for cooling values
dataset_Cool = np.zeros((len_z,len_nHenH,len_T,len_nh))

for i in range(0,len_z):

    #For each table it is stored z and its cooling values

    dataset_z[i] = np.array(h5py.File(testarray[i], "r")[f"{header_name}/{z}"]) 
       
    dataset_Cool[i,:,:,:] = np.array(h5py.File(testarray[i], "r")[f"{group_name}/{NetCooling}"])

    #Close each file once saved
    h5py.File(testarray[i], "r").close


print('Shape z: ', dataset_z.shape)

#Correct dimensions
dataset_z=np.squeeze(dataset_z,axis=1)
print('Shape z: ', dataset_z.shape) #(49,)


# Build xyz coordinated from T and nH
xx,yy = np.meshgrid(dataset_T,dataset_nh,indexing='ij')
xyz = np.zeros((len(dataset_T)*len(dataset_nh),2),dataset_T.dtype) #Agafar variable len corresponent
xyz[:,0] = xx.flatten()
xyz[:,1] = yy.flatten()

print('Shape xyz: ',xyz.shape) #352*81,2 - pairs T-nH



#The points on the arrays must be ordered consecutively 
pointO=np.arange(xyz.shape[0])
print('pointO:', pointO.shape) #(28512,)

#Not necessary for nhe_nh
#He_h=np.array([dataset_nhe_nh],np.float32).T #If not transposed would be 1,11
#He_h=np.squeeze(He_h,axis=1)
#print('He_h:',He_h.shape) #(11,)

#z_red=dataset_z
#print('z:',z_red.shape) 


#The number of points matches the first dimension of xyz and lam
npoints=xyz.shape[0]

#Table with only one partition with the same number of points for elements
ptable=pyLOM.PartitionTable.new(1,npoints,npoints)
print(ptable)


#Create matrix with all the cooling values
lam=np.zeros((len_T*len_nh,len_nHenH,len_z))

#The groups generated for the diferent z are joined 
#lam must be (28512,11,49) and Cool_set (49,11,352,81)

for i in range(0,len_z):
    for j in range(0,len_nHenH):

      lam[:,j,i]=dataset_Cool[i,j,:,:].flatten(order='C')
      #print(i,j)

print(lam)
print(lam.shape)


##Creation pyLOM dataset

d=pyLOM.Dataset(xyz=xyz,ptable=ptable,order=pointO,point=True,
    #Varibles com diccionari, associated to the dimensions of the matrix lam
    vars={
        #S'associa la primera dimensió de lam (idim=0) com a dimensió 0 de lam
        # Sempre és el nombre de punts 

        'T' : {'idim':0, 'value':dataset_T},

        'nH' : {'idim':0, 'value':dataset_nh},


        'nHe_nH' : {'idim':0, 'value':dataset_nhe_nh},

        'z' : {'idim':1, 'value':dataset_z},
        },

        lam = {'ndim':1, 'value':lam}   #Camp escalar és 1 perquè és 1 
    )

print(d)


##Store dataset
d.save(OUTFILE,mode='w') #w to overwrite previous

"""

## Parameters 
INFILE = "/home/bsc/bsc320267/scratch/Cooling_tables/Split/z_0.000_HeatingOnly.hdf5" #"/home/bsc/bsc320267/scratch/Cooling_tables/KS19/z_[0-9]????.hdf5"  #"/home/bsc/bsc320267/scratch/Cooling_tables/Split/z_0.000_CoolingOnly.hdf5"

OUTFILE = "/home/bsc/bsc320267/scratch/datasets/New_datasets/KS19/MetalFree_dataset_0.000_Heat.h5"

#"./MetalFree_dataset_TnH.h5" 



generate_dataset_MetalFree(INFILE, OUTFILE)


pyLOM.cr_info()


