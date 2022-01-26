## Copyright: Siddarth Achar
## University of Pittsburgh


import numpy as np
import dscribe
from ase.build import molecule
from dscribe.descriptors import SOAP
from ase.io import read, write
from ase import Atom, Atoms
import time
import pickle
from ase.io.cube import read_cube, write_cube
from sklearn.metrics import explained_variance_score, mean_squared_error
import py3Dmol

class deepcdp:
    
    def __init__(self, soap_def):
        '''
        An instance of deepcdp would have one constant function, which is the SOAP function (soap_def). Any model that is trained using data generated an instance of deepcdp will only work on data that is generated from this definition of soap.  
        This is helpful for cross-flatforming.
        '''
             
        self.soap_def=soap_def
    
    def get_NoPoints_spacing_matrix(path):
        file=open(path,'r')
        lines = file.readlines()
        noPoints = (int(lines[3].split()[0]), int(lines[4].split()[0]), int(lines[5].split()[0]))
        spacing_matrix = np.array([lines[3].split()[1:],
                                   lines[4].split()[1:],
                                   lines[5].split()[1:]])
#         self.noPoints=noPoints
#         self.spacing_matrix=spacing_matrix.astype(float)
        
        return noPoints, spacing_matrix.astype(float)
    
    def calc_diffVolume(self):
        x_cell=self.spacing_matrix[:,0]
        y_cell=self.spacing_matrix[:,1]
        z_cell=self.spacing_matrix[:,2]
        diffV=np.dot(x_cell,np.cross(y_cell,z_cell))
        
    def create_box(self, bounds=None, cell_spacing=None, gamma=None, sample_cubeFile=None, stepX=1, stepY=1, stepZ=1):
        '''
        This function generates the grid points in a box based on the shape of the cell
        that is defined. The function in NOT generalized to ALL cell shapes. The input 
        parameters are:
        gamma: Gamma angle formed
        *All s*_b denote lattice/ cell parameters. Input in Angstroms*
        s11_b: XX
        s21_b: XY
        s22_b: YY
        s33_b: ZZ
        x_ini: Initial X (number of points) 
        x_fin: Final X
        y_ini: Initial Y
        y_fin: Final Y
        z_ini: Initial Z
        z_fin: Final Z
        stepX: Steps along X direction. Used to reduce number of grid points
        stepY: Steps along Y direction
        stepZ: Steps along Z direction

        TODO: Write a FULLY generalized function to read 

        '''
        if sample_cubeFile is not None:
            bounds, self.cell_spacing=deepcdp.get_NoPoints_spacing_matrix(sample_cubeFile)
            self.bounds=bounds
            x_fin, y_fin, z_fin=self.bounds
            x_ini, y_ini, z_ini=[0,0,0]
            s11_b=self.cell_spacing[0,0]
            s21_b=self.cell_spacing[1,0]
            s22_b=self.cell_spacing[1,1]
            s33_b=self.cell_spacing[2,2]
            if gamma==None:
                print('pass gamma (in degrees) as arg in function')
                exit
            # Here bounds include the entire box. If you need smaller cell, then define by yourself
        else:
            self.bounds=bounds
            x_ini, x_fin=self.bounds[0]
            y_ini, y_fin=self.bounds[1]
            z_ini, z_fin=self.bounds[2]
            gamma, s11_b, s21_b, s22_b, s33_b = self.cell_spacing
        
        gamma_rad = np.pi*gamma/180
        Bohr2Ang = 0.529177
        s11 = s11_b * Bohr2Ang
        s21 = s21_b * Bohr2Ang
        s22 = s22_b * Bohr2Ang
        s33 = s33_b * Bohr2Ang

        sinGamma = np.sin(gamma_rad)
        cosGamma = np.cos(gamma_rad)
        s_sqrt = np.sqrt(s21**2 + s22**2)  # Look at the equation

        # TODO: Is there a more general expression for this?
        box = []

        for i in np.arange(x_ini,x_fin,stepX):
            for j in np.arange(y_ini,y_fin,stepY):
                for k in np.arange(z_ini,z_fin,stepZ):
                    x = i*s11 + s_sqrt*cosGamma*j     #Had to change the cos to sin
                    y = s_sqrt*sinGamma*j             #Had to change the sin to cos
                    z = k*s33
                    box.append((x,y,z))
        self.box=box

#         return box
    
    def cube2xyz(self, cubepath, outpath):
        '''
        Function that takes in cube files and converts then to xyz files.
        '''
        dictCube_test = read_cube(cubepath)   #Reads to a dict
        center_test = dictCube_test['atoms']     #Atom centers from ase Atoms format
        write(outpath,center_test,format='xyz')
        
    def view_py3Dmol(self,xyzfile=None, cubefile=None, c='grey',show=True):
        
#         print(self.xyzview)
        if 'xyzview' not in dir(self):
            self.xyzview = py3Dmol.view(width=400,height=400)
        
        if xyzfile is not None:
            xyz_mol=open(xyzfile).read()
            self.xyzview.addModel(xyz_mol,'xyz')
            self.xyzview.setStyle({'stick':{}})
            
        if cubefile is not None:
            cube_read=open(cubefile).read()
            self.xyzview.addVolumetricData(cube_read,'cube',{'isoval': 0.03,"color":c,'opacity':0.75,'wireframe':True})
    
        if show==True:
            self.xyzview.zoomTo()
            self.xyzview.show()
        
    def generate_cube_data(self, samp, Cubefile_lambda, ase_atoms_sample, savedata_path=None):
        
        rho=[]
        j=0
        for i in samp:
            print(f'Read file: {Cubefile_lambda(i)}')
            dictCube = read_cube(Cubefile_lambda(i))           #Reads to a dict
            center = dictCube['atoms'].get_positions()   # Atom centers from ase Atoms format
            rho_inter = dictCube['data']
            rho_i_trim=rho_inter.ravel().reshape(-1,1)                                                    #Flattening the 3D matrix of densities
            rho_i = list(map(float,rho_i_trim))                 #Just to change the change and map to a list
        #     print(rho_i)
            if j == 1:
                ase_atoms_sample.set_positions([*center])            # Change the atomic positions for each cube file
                soap_vec = self.soap_def.create(ase_atoms_sample, positions=self.box)
                X_i = soap_vec
                X_train = np.vstack([X_train, X_i])
            if j == 0:
                ase_atoms_sample.set_positions([*center])
                soap_vec = self.soap_def.create(ase_atoms_sample, positions=self.box)
                print('SOAP is of dimension:',soap_vec.shape)
                X_train = soap_vec
                j = 1
            rho += rho_i # List addition
        rho = np.array(rho)
        self.trainX=X_train
        self.trainY=rho
        
        if savedata_path is not None:
            np.savetxt(f'{savedata_path}/trainingX.dat', X_train)
            np.savetxt(f'{savedata_path}/trainingY.dat', rho)
    

    def test_1cube_full_box(self, Cubefile, model, save_pred_path=None):
        
        dictCube_test = read_cube(Cubefile)                #Reads to a dict
        center_test = dictCube_test['atoms']   # Atom centers from ase Atoms format
        rho_test = dictCube_test['data']
        rho_test = rho_test.ravel().reshape(-1,1)
        soap_test_vec = self.soap_def.create(center_test, positions=self.box)
        rho_pred = model.predict(soap_test_vec)
        if save_pred_path is not None:
            bounds, cell_spacing=deepcdp.get_NoPoints_spacing_matrix(Cubefile)
            x_fin, y_fin, z_fin=bounds
            rho_pred_resp = rho_pred.reshape(x_fin, y_fin, z_fin)
            output=open(save_pred_path,'w+')
            write_cube(output,center_test,rho_pred_resp,origin=self.box[0])           
        return rho_test.T[0], rho_pred 
    
    def change_spacing_cube(cubefile, cell_spacing):
        '''
        The write cube function by ASE automatically sets the spacing
        between grid points based on the input. For smaller sections of
        the system, we would need to smaller cell spacing that what ASE 
        assigns. This function rewrites the cube file for the spacing 
        that we specify. 
        '''
        file=open(cubefile,'r')
        file_read = file.readlines()
        x_line=file_read[3].split()
        y_line=file_read[4].split()
        z_line=file_read[5].split()
        x_line[1:]=cell_spacing[0,:].astype(str)
        y_line[1:]=cell_spacing[1,:].astype(str)
        z_line[1:]=cell_spacing[2,:].astype(str)
        file_read[3] = "    ".join(x_line) + "\n"
        file_read[4] = "    ".join(y_line) + "\n"
        file_read[5] = "    ".join(z_line) + "\n"
        fout=open(cubefile,'w')
        for elements in file_read:
            fout.write(elements)
        fout.close()
        
    # Octant 1
    def predict_soap_write(self, gamma, x_i, x_f, y_i, y_f, z_i, z_f, outpath,
                      ase_file_atomic,model,cell_spacing=None, stepX=1, stepY=1, stepZ=1):
        '''
        This function does the following:
        1. Creates a box with points based on the cell spacing and initial 
        and final x, y and z points.
        2. Generates a SOAP vector using the points created above.
        3. Performs prediction for the set of points.
        4. Writes the prediction to outpath as a cube file.
        5. If specified, changes the cell spacing based on user input
        '''
        gamma_rad = np.pi*gamma/180
        Bohr2Ang = 0.529177
#         print(cell_spacing)
        s11 = cell_spacing[0,0] * Bohr2Ang
        s21 = cell_spacing[1,0] * Bohr2Ang
        s22 = cell_spacing[1,1] * Bohr2Ang
        s33 = cell_spacing[2,2] * Bohr2Ang

        sinGamma = np.sin(gamma_rad)
        cosGamma = np.cos(gamma_rad)
        s_sqrt = np.sqrt(s21**2 + s22**2)  # Look at the equation

        # TODO: Is there a more general expression for this?
        box_oct = []

        for i in np.arange(x_i,x_f,stepX):
            for j in np.arange(y_i,y_f,stepY):
                for k in np.arange(z_i,z_f,stepZ):
                    x = i*s11 + s_sqrt*cosGamma*j     #Had to change the cos to sin
                    y = s_sqrt*sinGamma*j             #Had to change the sin to cos
                    z = k*s33
                    box_oct.append((x,y,z))

        soap_oct=self.soap_def.create(ase_file_atomic, positions=box_oct)
        rho_pred_oct=model.predict(soap_oct)

        out=open(outpath,'w+')
        rho_test_resp = rho_pred_oct.reshape(x_f-x_i, y_f-y_i ,z_f-z_i)
        write_cube(out,ase_file_atomic,rho_test_resp,origin=box_oct[0])

        if cell_spacing is not None:
            deepcdp.change_spacing_cube(outpath,cell_spacing)

        print(f"Generated {outpath}")

        del box_oct, soap_oct, rho_pred_oct, rho_test_resp
