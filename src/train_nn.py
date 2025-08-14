from sklearn.model_selection import learning_curve
from svd_dataset import SVD
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from extract_mesh import *
import pyvista as pv
import h5py




def getWallProperty(property, vtkfile):
    #vtkfile = '/home/ppiper/Dropbox/local/thermal_nozzle/src_air/yeom/outputs/fluid.vtk'
    output = pv.read(vtkfile)
    edges = output.extract_feature_edges()
    wall_idx = np.where(edges['Heat_Flux'] != 0)[0]
    property_at_wall = edges[property][wall_idx]
    y = edges.points[wall_idx,1]
    x = edges.points[wall_idx,0]
    return property_at_wall, x, y

def get_wall_idx(vtkfile):
    output = pv.read(vtkfile)
    #edges = output.extract_feature_edges()
    edges = output
    wall_idx = np.where(edges['Heat_Flux'] != 0)[0]
    return wall_idx


def saveMesh(vtkfile):
    vtk = pv.read(vtkfile)
    vtk.clear_data()
    dot_idx = vtkfile.rfind('.')
    vtkfile_mesh = vtkfile[:dot_idx]+'_mesh.vtk'
    vtk.save(vtkfile_mesh)

def plotMesh(meshfile, arr, property):
    p = pv.Plotter()
    p.enable()
    p.enable_anti_aliasing()

    mesh = pv.read(meshfile)
    mesh[property] = arr[:]
    mesh.set_active_scalars(property)
    p.add_mesh(mesh, opacity=0.85, render=True, cmap='plasma')
    p.add_mesh(mesh.contour(), color="white", line_width=2, render=True, cmap='plasma')

    p.set_viewup([0, 1, 0])
    #p.fly_to([5,0,0])

    #p.set_position([5.0, -0.01, 7.5])
    p.window_size = [1280,480]
    #p.save_graphic('./annSU2_'+str(pod.modes.shape[1])+'_modes.pdf')
    p.show(interactive_update=False, auto_close=True)
    #p.show()

def subtractMean(X):
    mean = np.mean(X, axis=1)
    for i, col in enumerate(X[0,:]):
        X[:,i] = X[:,i] - mean
    return X, mean

def addMean(X, mean):
    for i, col in enumerate(X[0,:]):
        X[:,i] = X[:,i] + mean
    return X

def normalize(data, bounds, min=None, max=None):
    if min is None and max is None:
        min = np.min(data)
        max = np.max(data)
    normdata = (bounds[1]-bounds[0])*(data - min)/(max-min) + bounds[0]
    return normdata, min, max

def denormalize(data, min, max, bounds):
    denormdata = (data - bounds[0])*(max-min)/(bounds[1]-bounds[0]) + min
    return denormdata

def extractPrimitives(snap):
    L=int(len(snap)/3)
    p = snap[0:L]
    rho = snap[L:2*L]
    Mach = snap[2*L:3*L]
    #return p, rho, Mach
    return snap[:], snap[:], snap[:]

def plotSU2(snapshot, primitive, mesh='../solution_mesh.vtk'):
    p, rho, Mach = extractPrimitives(snapshot)
    prim = {'p':p,
            'rho':rho,
            'Mach':Mach}
    plotMesh(mesh, prim[primitive])
    
def plotQ1D(snapshot, primitive):
    p, rho, Mach = extractPrimitives(snapshot)
    prim = {'p':p,
                 'rho':rho,
                 'Mach':Mach}
    xL = np.linspace(0, 1, len(p))

    plt.plot(xL, prim[primitive])

def get_model ( num_inputs , num_outputs , num_layers , num_neurons ):

    # Input layer
    ph_input = tf.keras.Input( shape =( num_inputs ,) ,name='input_placeholder')
    # Hidden layers
    hidden_layer = tf.keras.layers.Dense ( num_neurons , activation ='tanh')( ph_input )
    for layer in range ( num_layers ):
        hidden_layer = tf.keras.layers.Dense ( num_neurons , activation ='tanh')( hidden_layer )


    # Output layer
    output = tf.keras.layers.Dense ( num_outputs , activation ='linear',name='output_value')( hidden_layer)
    model = tf.keras.Model ( inputs =[ ph_input ], outputs =[ output ])
    # Optimizer
    my_adam = tf.keras.optimizers.Adam()
    # Compilation
    model.compile ( optimizer =my_adam , loss ={ 'output_value': 'mean_squared_error'})
    return model

def plotResults(idx, snapIn, snapRec2, data, conjunto):
    pv.set_plot_theme("document")
    
    print('Q1D input reconstructed in reduced space')
    p, rho, MachSVD = extractPrimitives(snapIn)
    plotQ1D(snapIn, 'Mach')
    plt.show()

    print('Actual NN results for SU2 reconstruction')
    p, rho, Mach2 = extractPrimitives(snapRec2)

    mesh = pv.read('../'+data+'/'+conjunto+'/SU2/'+str(idx+1)+'/outputs/solution.vtk')
    mesh.set_active_scalars('Mach')
    Mach = mesh['Mach']
    mesh2 = mesh.reflect((0, -1, 0), point=(0, 0, 0))
    mesh2['Mach'] = Mach2
    mesh = mesh + mesh2
    mesh.set_active_scalars('Mach')
    
    p = pv.Plotter()
    p.enable()
    p.enable_anti_aliasing()

    p.add_mesh(mesh, opacity=0.85, render=True, cmap='plasma')
    #p.add_mesh(mesh.contour(), color="white", line_width=2, render=True, cmap='plasma')

    p.set_viewup([0, 1, 0])
    p.fly_to([5,0,0])

    p.set_position([5.0, -0.01, 7.5])
    p.window_size = [1280,480]
    #p.save_graphic('./annSU2_'+str(pod.modes.shape[1])+'_modes.pdf')
    p.show(interactive_update=False, auto_close=True)
    #p.show()

    xLSU2 = np.linspace(0,1,len(Mach.reshape(46,201)[0,4:197]))
    xLQ1D = np.linspace(0,1,len(MachSVD))
    plt.plot(xLSU2, Mach.reshape(46,201)[0,4:197], label='SU2')
    plt.plot(xLSU2, Mach2.reshape(46,201)[0,4:197], label='NN')
    plt.plot(xLQ1D, MachSVD, label='Q1D')
    plt.legend()
    plt.ylabel('Mach')
    plt.xlabel('x/L')
    plt.show()

def getSnapshots(data, conjunto, rank):
    q1d_val = SVD('../'+data+'/Q1D_'+conjunto+'.hdf5')
    q1d_val.normalize(bounds=[0,1])
    q1d_val.subtractMean()
    q1d_val.SVD()
    q1d_val.setRank(rank)

    su2_val = SVD('../'+data+'/SU2_'+conjunto+'.hdf5')
    su2_val.normalize(bounds=[0,1])
    su2_val.subtractMean()
    su2_val.SVD()
    su2_val.setRank(rank)

    return q1d_val, su2_val

def plotSVD(idx, q1d, su2):
    pass

def plotPredictions(idx, model, q1d, su2, data, conjunto, q1d_):
    L = (q1d.u.T @ q1d_.snapshots[:,idx:idx+1]).T
    #su2LnnIDX = model.predict(q1d.L[idx:idx+1,:])
    su2LnnIDX = model.predict(L)

    #su2LnnIDX = denormalize(su2LnnIDX, su2Min, su2Max, bounds)
    #q1dL = denormalize(q1d.L, q1dMin, q1dMax, bounds)

    snapRec = su2.reconstruct(su2LnnIDX)

    #snapIn = q1d.reconstruct(idx)
    snapIn = q1d_.getSnapshot(idx)

    plotResults(idx, snapIn, snapRec, data, conjunto)

def orderReduction(low_fidelity_data, high_fidelity_data, rank, data_split):
    from data_handler import sliceDataAlongAxis

    bounds = [0,1]


    lf_data = sliceDataAlongAxis(
        low_fidelity_data, data_split, 0)

    lf_roms = []
    for i, data_split in enumerate(lf_data):
        q1d = SVD(data_split)
        q1d.normalize(bounds=bounds)
        q1d.subtractMean()
        q1d.SVD()
        q1d.setRank(rank)
        lf_roms.append(q1d)

        if i == 0:
            plt.figure()
            q1d.plotEnergy(lw=0, marker='o', 
                        label='Rank={:.2f}, Accumlated Energy: {:.2f} %'.format(rank, q1d.acumEnergy()))
            plt.legend()
            plt.show()


    hf_data = sliceDataAlongAxis(
        high_fidelity_data, data_split, 0) 

    hf_roms = []
    for i, data_split in enumerate(hf_data):
        su2 = SVD( data_split)
        su2.normalize(bounds=bounds)
        su2.subtractMean()
        su2.SVD()
        su2.setRank(rank)
        hf_roms.append(su2)

        if i == 0:
            plt.figure()
            su2.plotEnergy(lw=0, marker='o', 
                        label='Rank={:.2f}, Accumlated Energy: {:.2f} %'.format(rank, su2.acumEnergy()))
            plt.legend()
            plt.show()

    print('END SVD')

    return lf_roms, hf_roms, rank

def trainNN_thickness(q1d, thickness, su2, rank):

    model = get_model(rank+1, rank, 5, rank*2)

    ntrain=139
    ntest=139+17
    nvalid=139+17


    q1d_train = np.concatenate([q1d.L[:ntrain,...], thickness[:ntrain,None]], axis=1)
    su2_train = su2.L[:ntrain,...]
   
    q1d_test = np.concatenate([q1d.L[ntrain:ntest,...], thickness[ntrain:ntest,None]], axis=1)
    su2_test = su2.L[ntrain:ntest,...]
    
    q1d_validation = np.concatenate([q1d.L[ntest:nvalid,...], thickness[ntest:nvalid,None]], axis=1)
    su2_validation = su2.L[ntest:nvalid,...]

    model.fit(q1d_train, su2_train, epochs=1000, batch_size=16, validation_data=(q1d_validation, su2_validation)) 


    return model

    # use the same su2 basis to reconstruct the model
    #plotPredictions(28, model, q1d, su2, 'dataVERYLARGE', 'validation', q1d_validation)

# def trainNN(q1d_train, su2_train, q1d_validation, su2_validation, q1d_rank, su2_rank):

#     model = get_model(q1d_rank, su2_rank, 10, su2_rank*10)

#     model.fit(q1d_train, su2_train, epochs=1000, batch_size=16, validation_data=(
#         q1d_validation, su2_validation)) 

#     return model


def trainNN(inputs_train, outputs_train, inputs_validation, outputs_validation, layers):
    model = get_model(*(layers))

    history = model.fit(inputs_train, outputs_train, epochs=1000, batch_size=16,
                        validation_data=(inputs_validation, outputs_validation))

    return model, history


def trainNN_scalars(q1d, su2):

    model = get_model(q1d.shape[1], su2.shape[1], 5, su2.shape[1])

    ntrain=139
    ntest=139+17
    nvalid=139+17


    q1d_train = q1d[:ntrain,...]
    su2_train = su2[:ntrain,...]
   
    q1d_test = q1d[ntrain:ntest,...]
    su2_test = su2[ntrain:ntest,...]
    
    q1d_validation = q1d[ntest:nvalid,...]
    su2_validation = su2[ntest:nvalid,...]

    model.fit(q1d_train, su2_train, epochs=1000, batch_size=16, validation_data=(q1d_validation, su2_validation)) 

    return model

def trainNN_thickness_embeded():
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/Q1D.hdf5', 'r') as f:
        domain_size = int(f['snapshots'][()].shape[0]/3)
        thickness_distribution = np.array([f['thickness'][()]]*domain_size )
        lfd = f['snapshots'][()]
        lfd = np.concatenate([lfd, thickness_distribution])
        

    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_fluid.hdf5', 'r') as f:
        hfd = f['snapshots'][()]
        thickness = f['thickness'][()]

    q1d, su2, rank = orderReduction(low_fidelity_data = lfd,
                                    high_fidelity_data = hfd, 
                                    rank = 50)

    model = trainNN(q1d, su2, rank)
    model.save('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/trained_nn_thickness_embeded')
    return model


def trainNN_only_temperature():
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/Q1D.hdf5', 'r') as f:
        domain_size = int(f['snapshots'][()].shape[0]/3)
        thickness_distribution = np.array([f['thickness'][()]]*domain_size )
        lfd = f['snapshots'][()][domain_size*1:domain_size*2]
        lfd = np.concatenate([lfd, thickness_distribution])
        
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_fluid.hdf5', 'r') as f:
        domain_size = int((f['snapshots'][()].shape[0]-160)/3)
        hfd = f['snapshots'][()][domain_size*1:domain_size*2]
        thickness = f['thickness'][()]

    q1d, su2, rank = orderReduction(low_fidelity_data = lfd,
                                    high_fidelity_data = hfd, 
                                    rank = 50)

    model = trainNN(q1d, su2, rank)
    model.save('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/trained_nn_only_temperature')
    return model

def trainNN_only_scalars():
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/Q1D.hdf5', 'r') as f:
        thickness = f['thickness'][()]
        p0in = f['p0in'][()]
        T0in = f['T0in'][()]
        lfd = np.concatenate([p0in[:,None], T0in[:,None], thickness[:,None]], axis=1)
        
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_fluid.hdf5', 'r') as f:
        domain_size = int((f['snapshots'][()].shape[0]-160)/3)
        hfd = f['snapshots'][()][domain_size*1:domain_size*2]
        thickness = f['thickness'][()]

    q1d, su2, rank = orderReduction(low_fidelity_data = lfd,
                                    high_fidelity_data = hfd, 
                                    rank = 50)
    #q1d = lfd

    model = trainNN_scalars(lfd, su2.L)
    model.save('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/trained_nn_only_scalars')
    return model

def main():
    
    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/Q1D.hdf5', 'r') as f:
        lfd = f['snapshots'][()]

       

    with h5py.File('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/SU2_fluid.hdf5', 'r') as f:
        hfd = f['snapshots'][()]
        thickness = f['thickness'][()]

    q1d_rom, su2_rom, rank = orderReduction(low_fidelity_data = lfd,
                                    high_fidelity_data = hfd, 
                                    rank = 50)

    #model = trainNN_thickness(q1d, thickness, su2, rank)
    #model.save('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/trained_nn_thickness')


    nn_model = trainNN(q1d_rom, su2_rom, rank)
    nn_model.save('/home/ppiper/Dropbox/local/ihtc_nozzle/data/doe_lhs_N200/trained_nn')

    return nn_model
if __name__ == '__main__':

    nn_model = main()
    #trainNN_thickness_embeded()
    #trainNN_only_temperature()
    #trainNN_only_scalars()