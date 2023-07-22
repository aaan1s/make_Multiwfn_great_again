'''
unst - Using Generate_BCPlistEZ
Notes:
    -sysdiv=n/m/p/i N/i,j,k/i,j      Ex. n10 - first 10 atoms; m0,1,2 - atoms 1,2,3 will be CoreAtoms; p0,1; i1
                                         where 1 is atomic number in needed molecule
    -ifunc=x/u number                Ex. x1 for edensity
    -imode=use_normals               Ex. Y
    -gensurf=x/u number              Ex. x1 for edensity
    -isocut=                         Ex. 0.0
    -useoldsurfs=y/n/c               needs temp_* CoreCluster_BCPs* if y, regenerates CoreCluster_BCPs if c
    

    use_normals = S  - square of the surface
                = Y  - RnRho/R**2
                = N  - Rho/R
                = Y+R - norm*r(at-surf)*Rho
                = Y+Re ~ Y+R + norm*r(at-surf)*Rho) for all other atoms in the system
                                                      with the surface of chosen atom(s)
    
    
    defaults (if not mentioned): -sysdiv=n1 -ifunc=x1 -imode=Y -gensurf=x1 -isocut=0.0 -useoldsurfs=n


Changelog:
          3.0   The code was compleltely re-workerd!
          3.1   Added 'Man' variant for imode
          3.2   Added 'RedY' variant for imode
          3.3   Added new variant to remove oscillations
          3.4   Minor bugfixes
          3.5   Have the normals reverse algorithm repaired (?)
          3.6   The alternative way to define Inner/Outer atoms has been added
          3.7   Ability to limit the surface with isosurface value + surface area output
          3.8 - skipped
          3.9   Added ability to choose the function for surface generation
          4.0   Reworked input, tested previous (3.9) version, using new dnc2all library,
                returned cutoff test to chop turning rays, 
                added one more condition for oscillation cutoff (np.linalg.norm(vec1-vec2)<DistBtwPoints*1.3)
          4.1   Added compatibility with new variant of surface generation (bisection) - MultiwfnNSG
                stderr in Run_MWFN and class IAS is transfered to /dev/null
                Other cosmetic features were also added
                Modified Triangulation code (vector angle + area control)
                Added more points for CPs search (8000)
          4.2   Added support for *.wfx files
                Added -useoldsurfs=c, regenerates CP list, if CoreCluster_BCPs* present, checks it for missed CP
                    and generates surfaces for them. Adds new CPs to old CoreCluster_BCPs* file + generates 
                    CoreCluster_BCPs_new.txt file
                Added "Y_v2" variant = r1n1rho/|r1| + r2n2rho/|r2|
                Added "Dipole" variant = [rho_A,rho_B,ir1,ir2,dmu1,dmu2,AB,AdB,BdA,Total,R]
                Added "Y+Coul" variant = all pair (rho_A-ZA)(rho_B-ZB)/R and SurfInt/R (Y variant) [AB,SurfInt,Total,SurfInt/R]
                Added new function in GenSurp() - IntegrRhoInBasin.
                Corrected error when GenSurp module culd not be loaded (x=GenSurp() now GenSurp=GenSurp())
                Added "myTest" variant (can be deleted; also delet iatcharge, oatcharge) - now its IV test
                Added outfilename instead of inputfile[:-5] - better support for wfx wfn files
                Added "MSeries" variant: M_A = rho_A*r_A*n_A + lapl(rho_A)*r_A*n_A/|r_A|^2; Out= [M_A+M_B, M_A+M_B/R_AB, R_AB, S, mu_A, alpha_A, mu_B, alpha_B]

          4.3   Added sysdiv mode "i" - divides molecule using my old script for system split into moles (repaired version)
                Created new function Generate_BCPlistEZ (using standart points from MWFN Topology) + gen self.CCP
                Corrected the surf cutoff process:
                    (I - using CCP from Generate_BCP to remove closest surface points - sometimes they form a shitball)

                Y mode now has adequate output (at1 at2 I I/R R SurfArea) 
                Generate_BCPlistEZ uses each atom for sphere centring

Also: contains minor changes to filenames!!
      surface name is now outfilename+ '_surp_' + BCPsInfo[i][0] + '_' + BCPsInfo[i][1] + '.txt'
      temp files is now 'temp_'+outfilename+'_'+str(nCP)+'.{0}'   

'''

#######################     Imports     ##############################
import os
import sys
import numpy as np
import shutil
import re
import copy

#######################     MY CLASS    ##########################

class GenSurp():

    nproc = '40'

    ##### different little functions #####
    # my old reduction function
    def reduction(self,x):
        reduced_elem = []
        for i in x:
            if i not in reduced_elem:
                reduced_elem.append(i)
        return reduced_elem

    # my new function for finding the difference vector (returns list of coords in given units!!!)
    def VectDifference(self,XE, YE, ZE, XB, YB, ZB):
        [XE, YE, ZE, XB, YB, ZB] = [float(x) for x in [XE, YE, ZE, XB, YB, ZB]]
        XD = XE - XB
        YD = YE - YB
        ZD = ZE - ZB
        return [XD, YD, ZD]

    # my old function for locating the center of vector
    def Vector_Center(self, XM, YM, ZM, XS, YS, ZS):
        centerX = (XS + (XM - XS) * 0.5)
        centerY = (YS + (YM - YS) * 0.5)
        centerZ = (ZS + (ZM - ZS) * 0.5)
        return [centerX, centerY, centerZ]

    # my new standard distance finder (in Angstroms, i.e. Bohr radii --> A!!!)
    def calculate_distance(self, a, b):
        '''
        Returns distance between a and b
        '''
        return (sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5)

    # my new standard pseudo-covalent bond path intramolecular distance determination
    def Intramol(self,x, y):
        CUTOFF_DIST = float(self.dnc2all[x][1] / 2 + self.dnc2all[y][1] / 2)
        return CUTOFF_DIST

    # my new standard vdW bond path intermolecular distance determination
    def BONDii(self, x, y):
        CUTOFF_DIST = round(self.dnc2all[x][1] + self.dnc2all[y][1])+4.5
        return CUTOFF_DIST

    ##### different dictionaries #####
    # dictionary of (nuclear_charge : Nucleus)
    dat2num = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                    'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 20: 'Ca', 'Sc': 21,
                    'Ti': 22,
                    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 31: 'Ga',
                    'Ge': 32,
                    'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 41: 'Nb',
                    'Mo': 42,
                    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 51: 'Sb',
                    'Te': 52,
                    'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 61: 'Pm',
                    'Sm': 62,
                    'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 71: 'Lu',
                    'Hf': 72,
                    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 81: 'Tl',
                    'Pb': 82,
                    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 91: 'Pa',
                    'U': 92,
                    'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                    'No': 102,
                    'Lr': 103}
    # dictionary of (nuclear_charge : Nucleus, vdW radii in Angstroms)
    dnc2all = {1:['H',1.09,0.23,1.00794,0.99609375,0.9765625,0.80078125],
    2:['He',1.40,1.50,4.002602,0.72265625,0.82421875,0.9296875],
    3:['Li',1.82,1.28,6.941,0.7421875,0.7421875,0.7421875],
    4:['Be',2.00,0.96,9.012182,0.7421875,0.7421875,0.7421875],
    5:['B',2.00,0.83,10.811,0.625,0.234375,0.234375],
    6:['C',1.70,0.68,12.0107,0.328125,0.328125,0.328125],
    7:['N',1.55,0.68,14.0067,0.1171875,0.5625,0.99609375],
    8:['O',1.52,0.68,15.9994,0.99609375,0.0,0.0],
    9:['F',1.47,0.64,18.998403,0.99609375,0.99609375,0.0],
    10:['Ne',1.54,1.50,20.1797,0.72265625,0.82421875,0.9296875],
    11:['Na',2.27,1.66,22.98977,0.7421875,0.7421875,0.7421875],
    12:['Mg',1.73,1.41,24.305,0.7421875,0.7421875,0.7421875],
    13:['Al',2.00,1.21,26.981538,0.7421875,0.7421875,0.7421875],
    14:['Si',2.10,1.20,28.0855,0.82421875,0.82421875,0.82421875],
    15:['P',1.80,1.05,30.973761,0.99609375,0.546875,0.0],
    16:['S',1.80,1.02,32.065,0.99609375,0.9609375,0.55859375],
    17:['Cl',1.75,0.99,35.453,0.0,0.99609375,0.0],
    18:['Ar',1.88,1.51,39.948,0.72265625,0.82421875,0.9296875],
    19:['K',2.75,2.03,39.0983,0.7421875,0.7421875,0.7421875],
    20:['Ca',2.00,1.76,40.078,0.7421875,0.7421875,0.7421875],
    21:['Sc',2.00,1.70,44.95591,0.7421875,0.7421875,0.7421875],
    22:['Ti',2.00,1.60,47.867,0.7421875,0.7421875,0.7421875],
    23:['V',2.00,1.53,50.9415,0.7421875,0.7421875,0.7421875],
    24:['Cr',2.00,1.39,51.9961,0.7421875,0.7421875,0.7421875],
    25:['Mn',2.00,1.61,54.938049,0.7421875,0.7421875,0.7421875],
    26:['Fe',2.00,1.52,55.845,0.7421875,0.7421875,0.7421875],
    27:['Co',2.00,1.26,58.9332,0.7421875,0.7421875,0.7421875],
    28:['Ni',1.63,1.24,58.6934,0.7421875,0.7421875,0.7421875],
    29:['Cu',1.40,1.32,63.546,0.99609375,0.5078125,0.27734375],
    30:['Zn',1.39,1.22,65.409,0.7421875,0.7421875,0.7421875],
    31:['Ga',1.87,1.22,69.723,0.7421875,0.7421875,0.7421875],
    32:['Ge',2.00,1.17,72.64,0.7421875,0.7421875,0.7421875],
    33:['As',1.85,1.21,74.9216,0.7421875,0.7421875,0.7421875],
    34:['Se',1.90,1.22,78.96,0.7421875,0.7421875,0.7421875],
    35:['Br',1.85,1.21,79.904,0.7421875,0.5078125,0.234375],
    36:['Kr',2.02,1.50,83.798,0.72265625,0.82421875,0.9296875],
    37:['Rb',2.00,2.20,85.4678,0.7421875,0.7421875,0.7421875],
    38:['Sr',2.00,1.95,87.62,0.7421875,0.7421875,0.7421875],
    39:['Y',2.00,1.90,88.90585,0.7421875,0.7421875,0.7421875],
    40:['Zr',2.00,1.75,91.224,0.7421875,0.7421875,0.7421875],
    41:['Nb',2.00,1.64,92.90638,0.7421875,0.7421875,0.7421875],
    42:['Mo',2.00,1.54,95.94,0.7421875,0.7421875,0.7421875],
    43:['Tc',2.00,1.47,98.0,0.7421875,0.7421875,0.7421875],
    44:['Ru',2.00,1.46,101.07,0.7421875,0.7421875,0.7421875],
    45:['Rh',2.00,1.45,102.9055,0.7421875,0.7421875,0.7421875],
    46:['Pd',1.63,1.39,106.42,0.7421875,0.7421875,0.7421875],
    47:['Ag',1.72,1.45,107.8682,0.99609375,0.99609375,0.99609375],
    48:['Cd',1.58,1.44,112.411,0.7421875,0.7421875,0.7421875],
    49:['In',1.93,1.42,114.818,0.7421875,0.7421875,0.7421875],
    50:['Sn',2.17,1.39,118.71,0.7421875,0.7421875,0.7421875],
    51:['Sb',2.00,1.39,121.76,0.7421875,0.7421875,0.7421875],
    52:['Te',2.06,1.47,127.6,0.7421875,0.7421875,0.7421875],
    53:['I',1.98,1.40,126.90447,0.625,0.125,0.9375],
    54:['Xe',2.16,1.50,131.293,0.72265625,0.82421875,0.9296875],
    55:['Cs',2.00,2.44,132.90545,0.7421875,0.7421875,0.7421875],
    56:['Ba',2.00,2.15,137.327,0.7421875,0.7421875,0.7421875],
    57:['La',2.00,2.07,138.9055,0.7421875,0.7421875,0.7421875],
    58:['Ce',2.00,2.04,140.116,0.7421875,0.7421875,0.7421875],
    59:['Pr',2.00,2.03,140.90765,0.7421875,0.7421875,0.7421875],
    60:['Nd',2.00,2.01,144.24,0.7421875,0.7421875,0.7421875],
    61:['Pm',2.00,1.99,145.0,0.7421875,0.7421875,0.7421875],
    62:['Sm',2.00,1.98,150.36,0.7421875,0.7421875,0.7421875],
    63:['Eu',2.00,1.98,151.964,0.7421875,0.7421875,0.7421875],
    64:['Gd',2.00,1.96,157.25,0.7421875,0.7421875,0.7421875],
    65:['Tb',2.00,1.94,158.92534,0.7421875,0.7421875,0.7421875],
    66:['Dy',2.00,1.92,162.5,0.7421875,0.7421875,0.7421875],
    67:['Ho',2.00,1.92,164.93032,0.7421875,0.7421875,0.7421875],
    68:['Er',2.00,1.89,167.259,0.7421875,0.7421875,0.7421875],
    69:['Tm',2.00,1.90,168.93421,0.7421875,0.7421875,0.7421875],
    70:['Yb',2.00,1.87,173.04,0.7421875,0.7421875,0.7421875],
    71:['Lu',2.00,1.87,174.967,0.7421875,0.7421875,0.7421875],
    72:['Hf',2.00,1.75,178.49,0.7421875,0.7421875,0.7421875],
    73:['Ta',2.00,1.70,180.9479,0.7421875,0.7421875,0.7421875],
    74:['W',2.00,1.62,183.84,0.7421875,0.7421875,0.7421875],
    75:['Re',2.00,1.51,186.207,0.7421875,0.7421875,0.7421875],
    76:['Os',2.00,1.44,190.23,0.7421875,0.7421875,0.7421875],
    77:['Ir',2.00,1.41,192.217,0.7421875,0.7421875,0.7421875],
    78:['Pt',1.72,1.36,195.078,0.7421875,0.7421875,0.7421875],
    79:['Au',1.66,1.50,196.96655,0.99609375,0.83984375,0.0],
    80:['Hg',1.55,1.32,200.59,0.7421875,0.7421875,0.7421875],
    81:['Tl',1.96,1.45,204.3833,0.7421875,0.7421875,0.7421875],
    82:['Pb',2.02,1.46,207.2,0.7421875,0.7421875,0.7421875],
    83:['Bi',2.00,1.48,208.98038,0.7421875,0.7421875,0.7421875],
    84:['Po',2.00,1.40,290.0,0.7421875,0.7421875,0.7421875],
    85:['At',2.00,1.21,210.0,0.7421875,0.7421875,0.7421875],
    86:['Rn',2.00,1.50,222.0,0.72265625,0.82421875,0.9296875],
    87:['Fr',2.00,2.60,223.0,0.7421875,0.7421875,0.7421875],
    88:['Ra',2.00,2.21,226.0,0.7421875,0.7421875,0.7421875],
    89:['Ac',2.00,2.15,227.0,0.7421875,0.7421875,0.7421875],
    90:['Th',2.00,2.06,232.0381,0.7421875,0.7421875,0.7421875],
    91:['Pa',2.00,2.00,231.03588,0.7421875,0.7421875,0.7421875],
    92:['U',1.86,1.96,238.02891,0.7421875,0.7421875,0.7421875],
    93:['Np',2.00,1.90,237.0,0.7421875,0.7421875,0.7421875],
    94:['Pu',2.00,1.87,244.0,0.7421875,0.7421875,0.7421875],
    95:['Am',2.00,1.80,243.0,0.7421875,0.7421875,0.7421875],
    96:['Cm',2.00,1.69,247.0,0.7421875,0.7421875,0.7421875],
    97:['Bk',2.00,1.54,247.0,0.7421875,0.7421875,0.7421875],
    98:['Cf',2.00,1.83,251.0,0.7421875,0.7421875,0.7421875],
    99:['Es',2.00,1.50,252.0,0.7421875,0.7421875,0.7421875],
    100:['Fm',2.00,1.50,257.0,0.7421875,0.7421875,0.7421875],
    101:['Md',2.00,1.50,258.0,0.7421875,0.7421875,0.7421875],
    102:['No',2.00,1.50,259.0,0.7421875,0.7421875,0.7421875],
    103:['Lr',2.00,1.50,262.0,0.7421875,0.7421875,0.7421875]
    }


    ##################################      BODY        ##################################
    def __init__(self, inputfile):

        #### CENTERS generation, no convertation to Angstrom! [[Charge,x,y,z,AtomName],[...],...] ####
        if inputfile[-4:] == '.wfn':
            with open(inputfile) as file:
                self.CENTERS = []
                i=0
                for line in file:
                    if 'CHARGE' in line:
                        CENTER = []
                        i+=1
                        CENTER.append(int(float(line.split()[9])))
                        for x in line.split()[4:7]: CENTER.append(float(x))
                        CENTER.append(self.dnc2all[int(float(line.split()[9]))][0] + str(i))
                        self.CENTERS.append(CENTER)
            self.outfilename=inputfile[:-4]

        elif inputfile[-5:]=='.fchk':
            AT_NUMs = []
            COORDs = []
            with open(inputfile) as wfn:
                for line in wfn:
                    if 'Atomic numbers' in line:
                        for line in wfn:
                            if 'Nuclear charges' in line:
                                break
                            AT_NUMs = AT_NUMs + line.split()
                    if 'Current cartesian coordinates' in line:
                        for line in wfn:
                            if 'Number of symbols in' in line:
                                break
                            COORDs = COORDs + line.split()
            AT_NUM = [int(x) for x in AT_NUMs]
            COORDs2 = [COORDs[y:y + 3] for y in range(0, len(COORDs), 3)]
            self.CENTERS = []
            for i, x in enumerate(AT_NUM):
                self.CENTERS.append([x] + COORDs2[i])
            for i,Atom in enumerate(self.CENTERS):
                Atom[1] = float(Atom[1])
                Atom[2] = float(Atom[2])
                Atom[3] = float(Atom[3])
                Atom.append(self.dnc2all[Atom[0]][0] + str(i + 1))
            self.outfilename=inputfile[:-5]
        
        elif inputfile[-4:]=='.wfx':
            with open(inputfile) as file:
                self.CENTERS = []
                i=0
                for line in file:
                    if '<Nuclear Names>' in line:
                        Names = []
                        for line in file:
                            if '</Nuclear Names>' in line:
                                break
                            Names.append(line.split()[0])
                    if '<Atomic Numbers>' in line:
                        AN = []
                        for line in file:
                            if '</Atomic Numbers>' in line:
                                break
                            AN.append(float(line.split()[0]))
                    if '<Nuclear Cartesian Coordinates>' in line:
                        for i,line in enumerate(file):
                            if '</Nuclear Cartesian Coordinates>' in line:
                                break
                            self.CENTERS.append([AN[i]]+[float(x) for x in line.split()]+[Names[i]])
            self.outfilename=inputfile[:-4]
            
    #### Run Multiwfn
    def Run_MWFN(self, mytext, needout=False):
        with open('myprog.inp', 'w') as inp:
            inp.write('\n'.join(mytext) + '\n')
        if needout == True:
            os.system('{0} {1} < {2} > {3} 2>/dev/null'.format('./MultiwfnTest', inputfile, 'myprog.inp', 'myprog.out'))
        else:
            os.system('{0} {1} < {2} &>/dev/null'.format('./MultiwfnTest', inputfile, 'myprog.inp'))
        try:
            os.remove('myprog.inp')
        except FileNotFoundError:
            pass
        
    #### Calculate basin integrals
    def IntegrRhoInBasin(self,FalseAttractors=False):
        if FalseAttractors==False:
            print('Basin integration')
            text = ['1000', '10', nproc, '17', '1', '1', '3', '7', '2', '0.0004', '1']
        elif FalseAttractors==True:
            print('Basin integration - trouble with attractors')
            text = ['1000', '10', nproc, '17', '1', '1', '3', '3', '7', '2', '0.0004', '1']
        
        self.Run_MWFN(text,True)
        BasinsNames=[]
        BasinRhoIntegrals=[]
        with open('myprog.out') as out:
            for line in out:
                if 'Integrating in trust sphere' in line:
                    for line in out:
                        if 'Integration result' in line:
                            break
                        if 'Attractor' in line:
                            line=line.replace('(','').replace(')','')
                            BasinsNames.append(line.split()[-1]+line.split()[-2])
                if 'Integral(a.u.)' in line and 'Vol(Bohr^3)' in line:
                    for line in out:
                        if ' Sum of above integrals:' in line:
                            break
                        BasinRhoIntegrals.append(float(line.split()[1]))
        return BasinsNames,BasinRhoIntegrals

        
    ########################################  Topology branch ########################################
    

    #### Divide system on Core Atoms and Cluster (Outer) Atoms
    def DivideSys(self, DivSysParam):
        if DivSysParam[0] == 'm':
            InnerAt = DivSysParam[-1].split(',')
            i = 1
            self.CoreAtoms = []
            self.ClusterAtoms = self.CENTERS[:]
            for x in InnerAt:
                self.CoreAtoms.append(self.ClusterAtoms.pop(int(x) - i))
                i = i + 1
        elif DivSysParam[0] == 'p':
            InnerAt = DivSysParam[-1].split(',')
            self.CoreAtoms = [self.CENTERS[int(InnerAt[0]) - 1]]
            self.ClusterAtoms = [self.CENTERS[int(InnerAt[1]) - 1]]
        elif DivSysParam[0] == 'n':
            NumOfAt = int(DivSysParam[1])
            self.CoreAtoms = self.CENTERS[:NumOfAt]
            self.ClusterAtoms = self.CENTERS[NumOfAt:]
        elif DivSysParam[0] == 'i':
            CurrentAtoms=copy.deepcopy(self.CENTERS)
            CoreAtoms = [self.CENTERS[int(DivSysParam[1])-1]]
            CurrentAtoms.pop(int(DivSysParam[1])-1)
            i=0
            j=0
            while i<len(CoreAtoms):
                j = 0
                while j<len(CurrentAtoms):
                    if np.linalg.norm(np.array(CoreAtoms[i][1:-1])-np.array(CurrentAtoms[j][1:-1]))\
                            <(self.dnc2all[CoreAtoms[i][0]][1]+self.dnc2all[CurrentAtoms[j][0]][1]):
                        add=CurrentAtoms.pop(j)
                        CoreAtoms.append(add)
                        j=0
                    else:
                        j+=1
                i+=1
                
            self.CoreAtoms=CoreAtoms
            self.ClusterAtoms=CurrentAtoms

        print(len(self.CoreAtoms), 'in Core')
        print(len(self.ClusterAtoms), 'in Cluster')

    #### Generate BCP list
    def Generate_BCPlist(self, DivSysParam, gensurf):
        nproc = self.nproc
        self.DivideSys(DivSysParam)
        self.CCP=[]
        # GENERATING INITIAL GUESS FOR BCP COORDS
        print('Generating initial contacts')
        BONDS_CENTER = []
        for fatom in self.CoreAtoms:
            for satom in self.ClusterAtoms:
                if self.calculate_distance(fatom[1:-1], satom[1:-1])*0.52917720859 < self.BONDii(fatom[0], satom[0]):
                    BOND_CENTER = [fatom[4],satom[4]]
                    BOND_CENTER.append([str(x) for x in self.Vector_Center(fatom[1], fatom[2], fatom[3], satom[1], satom[2], satom[3])])
                    BONDS_CENTER.append(BOND_CENTER)

        MWFNsearch = []
        for center in BONDS_CENTER:
            TempMWFNsearch = []
            coord = ','.join(center[2])
            if gensurf[0]=='x':
                mytext = ['1000', '10', nproc, '2', '-11', gensurf[1], '6', '10', '3.4', '11', '8000', '1', coord, '0']
            elif gensurf[0]=='u':
                mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '2', '-11', '100', '6', '10', '3.4', '11', '8000', '1', coord, '0']
            self.Run_MWFN(mytext,True)
            with open('myprog.out') as sr:
                for line in sr:
                    if 'Summary' in line:
                        for line in sr:
                            if 'Totally find' in line:
                                break
                            TempMWFNsearch.append(line.split())
            MWFNsearch = MWFNsearch + TempMWFNsearch[1:]

        # TIME TO DELETE USELESS POINTS
        print('Deleting non-BCP points')
        MWFNpoints = []
        for line in MWFNsearch:
            x=[float(_) for _ in line[1:-1]]+[line[-1]]
            if x[-1]=='(3,-1)' and x[:-1] not in MWFNpoints:
                MWFNpoints.append(x[:-1])
            elif x[-1]=='(3,+3)' and x[:-1] not in self.CCP:
                self.CCP.append(x[:-1])

        # TIME FOR LOOKING FOR BOND PATHS
        print("Checking BCPs' bond paths")
        for x in MWFNpoints:
            CPcoord = [str(_) for _ in x]
            coord = ','.join(CPcoord)
            if gensurf[0]=='x':
                mytext = ['1000', '10', nproc, '2', '-11', gensurf[1], '1', coord, '2', '8', '-5', '2', '1', '2', '2']
            elif gensurf[0]=='u':
                mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '2', '-11', '100', '1', coord, '2', '8', '-5', '2', '1', '2', '2']
            self.Run_MWFN(mytext,True)
            with open('myprog.out') as sr:
                fNCP = []
                bNCP = []
                for line in sr:
                    if 'Path:     1' in line:
                        for line in sr:
                            if 'to' in line:
                                fNCP = line.split()[1:]
                            if 'The X/Y/Z' in line:
                                break
                    if 'Path:     2' in line:
                        for line in sr:
                            if 'to' in line:
                                bNCP = line.split()[1:]
                            if 'The X/Y/Z' in line:
                                break
            with open('MWFN_CPs_and_bp.txt', 'a') as cpbp:
                for Atom in self.CENTERS:
                    if self.calculate_distance(Atom[1:-1], [float(_) for _ in fNCP[:3]]) < 0.25:
                        cpbp.write(Atom[4]), cpbp.write(' ')
                    if self.calculate_distance(Atom[1:-1], [float(_) for _ in bNCP[:3]]) < 0.25:
                        cpbp.write(Atom[4]), cpbp.write(' ')
                cpbp.write(str(CPcoord[0])), cpbp.write(' '), cpbp.write(str(CPcoord[1])), cpbp.write(' '), \
                cpbp.write(str(CPcoord[2])), cpbp.write('\n')

        # DEALING WITH THE NON-INTERMOLECULAR BCPs
        with open('Core-Cluster_BCPs.txt', 'w') as hh:
            with open('MWFN_CPs_and_bp.txt') as cpbp:
                for line in cpbp:
                    for coa in self.CoreAtoms:
                        for cla in self.ClusterAtoms:
                            if line.split()[0] == coa[4] and line.split()[1] == cla[4]:
                                hh.write(line)
                            if line.split()[1] == coa[4] and line.split()[0] == cla[4]:
                                sortedline = line.split()[1] + ' ' + line.split()[0] + ' ' + line.split()[2] + ' ' + \
                                             line.split()[3] + ' ' + line.split()[4] + '\n'
                                hh.write(sortedline)

        for x in ['MWFN_CPs_and_bp.txt', 'myprog.inp', 'myprog.out']:
            try:
                os.remove(x)
            except FileNotFoundError:
                pass

        
    def Generate_BCPlistEZ(self, DivSysParam, gensurf):
        nproc = self.nproc
        self.DivideSys(DivSysParam)
        self.CCP = []
        MWFNsearch = []
        MWFNpaths = [] 
        
        if gensurf[0]=='x':
            mytext = ['1000', '10', nproc, '2', '-11', gensurf[1], '2', '3', '4', '5','6','-1','-9','-2', '3', '0.1','0','8','-5','1']
        elif gensurf[0]=='u':
            mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '2', '-11', '100', '2', '3', '4', '5','6','-1','-9','-2', '3', '0.1','0','8','-5','1']
        self.Run_MWFN(mytext,True)
        with open('myprog.out') as sr:
            for line in sr:
                if 'Summary' in line:
                    for line in sr:
                        if 'Totally find' in line:
                            break
                        if 'Coordinate' not in line:
                            MWFNsearch.append(line.split())
                if '--->' in line:
                    MWFNpaths.append([int(line.split()[3]),int(line.split()[7])])

        CoreNCPind=[]
        CoreNCPAt=[]
        ClustNCPind=[]
        ClustNCPAt=[]
        for CP in MWFNsearch:
            if CP[4]=='(3,+3)':
                self.CCP.append([float(x) for x in CP[1:-1]])
            for Atom in self.CENTERS:
                if np.linalg.norm(np.array([float(x) for x in CP[1:-1]])-np.array(Atom[1:-1]))<0.05 and CP[4]=='(3,-3)':
                    if Atom in self.CoreAtoms:
                        CoreNCPind.append(int(CP[0]))
                        CoreNCPAt.append(Atom[-1])
                    elif Atom in self.ClusterAtoms:
                        ClustNCPind.append(int(CP[0]))
                        ClustNCPAt.append(Atom[-1])
        
        InterMoleBCPs=[]
        for path1 in MWFNpaths:
            if path1[1] in CoreNCPind:
                for path2 in MWFNpaths:
                    if path1[0]==path2[0] and path2[1] in ClustNCPind:
                        InterMoleBCPs.append([CoreNCPAt[CoreNCPind.index(path1[1])],ClustNCPAt[ClustNCPind.index(path2[1])]]+[x for x in MWFNsearch[path1[0]-1][1:-1]])
        with open('Core-Cluster_BCPs.txt', 'w') as hh:
            for BCP in InterMoleBCPs:
                hh.write('{0[0]} {0[1]} {0[2]} {0[3]} {0[4]}\n'.format(BCP))


    def GenAllBCP(self):
        nproc = self.nproc
        self.DivideSys(DivSysParam)
        self.CCP = []
        MWFNsearch = []
        MWFNpaths = []

        if gensurf[0] == 'x':
            mytext = ['1000', '10', nproc, '2', '-11', gensurf[1], '2', '3', '4', '5', '-2', '3', '0.1', '0', '8', '-5',
                      '1']
        elif gensurf[0] == 'u':
            mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '2', '-11', '100', '2', '3', '4', '5', '-2', '3',
                      '0.1', '0', '8', '-5', '1']
        self.Run_MWFN(mytext, True)
        with open('myprog.out') as sr:
            for line in sr:
                if 'Summary' in line:
                    for line in sr:
                        if 'Totally find' in line:
                            break
                        if 'Coordinate' not in line:
                            MWFNsearch.append(line.split())
                if '--->' in line:
                    MWFNpaths.append([int(line.split()[3]), int(line.split()[7])])

        NCPind = []
        NCPAt = []
        for CP in MWFNsearch:
            if CP[4] == '(3,+3)':
                self.CCP.append([float(x) for x in CP[1:-1]])
            for Atom in self.CENTERS:
                if np.linalg.norm(np.array([float(x) for x in CP[1:-1]]) - np.array(Atom[1:-1])) < 0.05 and CP[4] == '(3,-3)':
                    NCPind.append(int(CP[0]))
                    NCPAt.append(Atom[-1])

        analyzed=[]
        InterMoleBCPs = []
        for path1 in MWFNpaths:
            for path2 in MWFNpaths:
                if path1[0] == path2[0] and path1!=path2 and path1 not in analyzed:
                    analyzed.append(path2)
                    InterMoleBCPs.append([NCPAt[NCPind.index(path1[1])], NCPAt[NCPind.index(path2[1])]] + [x for x in MWFNsearch[path1[0] - 1][1:-1]])

        with open('All_BCPs.txt', 'w') as hh:
            for BCP in InterMoleBCPs:
                hh.write('{0[0]} {0[1]} {0[2]} {0[3]} {0[4]}\n'.format(BCP))


    #### Generate Surfaces
    def Generate_Surfaces(self,DivSysParam='m', NumOfRays=100, NumOfPoints=200, DistBtwPoints=0.03, gensurf=['x','1'], isov=0.0):
        nproc = self.nproc
        DistBtwPoints=str(DistBtwPoints)

        BCPsInfo = []
        if DivSysParam!='a':
            self.Generate_BCPlistEZ(DivSysParam, gensurf)
        with open('Core-Cluster_BCPs.txt') as MWFNsr:
            for line in MWFNsr:
                BCPsInfo.append(line.split())

        for i in range(len(BCPsInfo)):
            print('Processing surface ', i+1, ' out of ', len(BCPsInfo))
            surf_name = self.outfilename + '_surp_' + BCPsInfo[i][0] + '_' + BCPsInfo[i][1] + '.txt'
            csurf_name = self.outfilename + '_surp_' + BCPsInfo[i][0] + '_' + BCPsInfo[i][1] + '_f.txt'
            coord = BCPsInfo[i][2] + ',' + BCPsInfo[i][3] + ',' + BCPsInfo[i][4]
            if gensurf[0] == 'u':
                mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '2', '-11', '100', '6', '10', '0.1', '1', coord, 
                          '0', '10', '3.0', '1', coord, '0', '-9', '2', '-3', '1', str(NumOfRays), '2',
                          str(NumOfPoints), '3', DistBtwPoints, '0', '10', '1', 'o 1']
            elif gensurf[0] == 'x':
                mytext = ['1000', '10', nproc, '2', '-11', gensurf[1], '6', '10', '0.1', '1', coord, '0', 
                          '10', '3.0', '1', coord, '0','-9', '2', '-3', '1',
                          str(NumOfRays), '2', str(NumOfPoints), '3', DistBtwPoints, '0', '10', '1', 'o 1']
            self.Run_MWFN(mytext)

            if os.path.exists(surf_name):
                os.remove(surf_name)
            os.rename('surpath.txt', surf_name)
            rays = []; ray=[]
            with open(surf_name) as surpath:
                for line in surpath:
                    if line.split()[0]=='Path':
                        if len(ray)>0: rays.append(ray[1:])
                        ray = []
                    ray.append(line.split())
                rays.append(ray[1:])


            # removing "tails" and "oscillations"
            print('Starting "oscillations" test')
            paths_length = [len(ray) for ray in rays]
            if isov==0.0:
                for c,ray in enumerate(rays):
                    for k in range(len(ray)):
                        for CP in self.CCP:
                            if np.linalg.norm(np.array(CP)-np.array([float(x) for x in ray[k][1:]]))<float(DistBtwPoints)*5:
                                print("Found shitball!")
                                paths_length[c]=k
                                break
                        try:
                            if ray[k][1:] == ray[k + 2][1:]:
                                paths_length[c]=k
                                #print('Ray ', c, ' !Oscillation:', ray[k], '==', ray[k + 2])
                                break
                        except IndexError:
                            pass

            else:
                iso = []
                with open("isotest.txt", 'w') as out:
                    out.write(str(sum(len(ray) for ray in rays))), out.write("\n")
                    for ray in rays:
                        for point in ray:
                            out.write("{0} {1} {2}\n".format(point[1], point[2], point[3]))
                
                if gensurf[0]=='x':
                    mytext = ['1000', '10', nproc, '5', gensurf[1], '100', "isotest.txt", "isotest.txt"]
                elif gensurf[0]=='u':
                    mytext = ['1000', '10', nproc, '1000', '2', gensurf[1], '5', '100', '100', "isotest.txt", "isotest.txt"]
                self.Run_MWFN(mytext)

                with open("isotest.txt") as out:
                    for line in out:
                        if len(line.split()) > 1:
                            iso.append(float(line.split()[-1]))
                # os.remove("isotest.txt")

                for c,ray in enumerate(rays):
                    for k in range(len(ray)):
                        try:
                            if ray[k][1:] == ray[k + 2][1:] or iso[c*NumOfPoints+k]<isov:
                                paths_length.append(k)
                                #print('Ray ', c, ' !Oscillation:', ray[k], '==', ray[k + 2], " or isov exclusion ", iso[c*NumOfPoints+k], "<", isov)
                                break
                        except IndexError:
                            if iso[c*NumOfPoints+k]<isov:
                                paths_length.append(k)
                                #print('Ray ', c, ' !Oscillation:', ray[k], '==', ray[k + 2], " or isov exclusion ",
                                #      iso[c * NumOfPoints + k], "<", isov)
                                break
                            else:
                                paths_length.append(len(ray))
                            pass
                        
            print('"oscillation" test finished! Writing choped surface')
            with open(csurf_name, 'w') as OUT:
                for j,ray in enumerate(rays):
                    Path = 'Path' + ' ' + str(j + 1)
                    OUT.write(str(Path)), OUT.write('\n')
                    for k,xyz in enumerate(ray):
                        OUT.write('{0[0]} {0[1]} {0[2]} {0[3]}\n'.format(xyz))
                        if k == paths_length[j]:
                            break


# REWORKED SIDE CLASS FOR IAS DIVISION
class IAS(GenSurp):
    def find_central_point(self,ps):
        '''
        Returns geometrical centre of n points
        For n points each coordinate is summed and divided their quantity
        Ex.: ps=[a,b] (a=[1,1,1], b=[1,2,3])
            then find_central_point(ps) will return
            [1,1.5,2]
        '''
        return [sum([_[0] for _ in ps]) / len(ps),
                sum([_[1] for _ in ps]) / len(ps),
                sum([_[2] for _ in ps]) / len(ps)]


    def calculate_triangle_square(self,ps):
        '''
        Returns square of triangle (Herons formula)
        ps = [a,b,c] (a,b,c - coordinates of points, forming the triangle
        '''
        a = self.calculate_distance(ps[0], ps[1])
        b = self.calculate_distance(ps[1], ps[2])
        c = self.calculate_distance(ps[2], ps[0])
        s = (a + b + c) / 2
        return (s * (s - a) * (s - b) * (s - c)) ** 0.5

    def calculate_square(self,ps):
        '''
        Returns sum of squares of (n-2) triangles if len(ps) == n
        '''
        triangles = [[ps[0], ps[i], ps[i + 1]] for i in range(1, len(ps) - 1)]
        return sum([self.calculate_triangle_square(_) for _ in triangles])

    def calculate_angle(self,a, b, c):
        '''
        Returns angle value
        (ordinary formula based on cos(alpha)=<a,b>/(|a||b|)
        '''
        ba = [ai - bi for ai, bi in zip(a, b)]
        bc = [ci - bi for ci, bi in zip(c, b)]
        cosine_angle = sum([bai * bci for bai, bci in zip(ba, bc)]) / (
                sum([_ ** 2 for _ in ba]) ** 0.5 * sum([_ ** 2 for _ in bc]) ** 0.5)
        angle = np.arccos(cosine_angle)
        return angle * 180 / np.pi

    def __init__(self, filepath):
        super(IAS, self).__init__(inputfile)
        '''
        It is created from Multiwfn output for IAS
        '''
        print('Submodule initialization')
        self.ias_divided = False
        self.paths = []
        path = []
        # reading points' data from surpath.txt
        with open(filepath, 'r') as inpf:
            text = [_.strip() for _ in inpf.readlines()]
        for line in text:
            if not line:
                continue
            if line[:4] == 'Path':
                if path:
                    self.paths.append(path)
                path = []
            else:
                path.append([float(_) for _ in line.split()[1:4]])
        if path:
            self.paths.append(path)
        # create the central point (BCP, i assume)
        self.centre = self.find_central_point([_[0] for _ in self.paths])
        # find step between path points
        # (i.e. calculating one of the constant values defined before the surface generation)
        self.step = self.calculate_distance(self.paths[0][0], self.paths[0][1])

    def check_lengths(self):
        '''
        Returns False if there are IAS paths of unequal length
        '''
        if len(set([len(_) for _ in self.paths])) > 1:
            return False
        else:
            return True

    def __repr__(self):
        ns = [len(_) for _ in self.paths]
        if len(set(ns)) > 1:
            return 'IAS containing {0} paths of unequal lengths: {1}'.format(len(ns), ', '.join(set(ns)))
        else:
            return 'IAS containing {0} paths {1} points each'.format(len(ns), ns[0])

    def __str__(self):
        ns = [len(_) for _ in self.paths]
        if len(set(ns)) > 1:
            return 'IAS containing {0} paths of unequal lengths: {1}'.format(len(ns), ', '.join(set(ns)))
        else:
            return 'IAS containing {0} paths {1} points each'.format(len(ns), ns[0])

    def paths2xyz(self, outpath):
        '''
        Writes xyz file for input IAS paths
        (format - overdokhuya strings '0 x y z')
        '''
        text = []
        for path in self.paths:
            for xyz in path:
                text.append('0  {0[0]: >13.10f} {0[1]: >13.10f} {0[2]: >13.10f}'.format(xyz))
        with open(outpath, 'w') as outf:
            outf.write('\n'.join(text) + '\n')


    def add2ias(self, a, b, c):
        '''
        a and b belong to one path
        Two variants
        If 'a' and 'c' are as close as nearest point in ray (i.e. 'b'):
            add central point of 'a','b','c' to ias_xyz
            calculate and add the square of triangle (=0?) to ias_S
        If 'a' and 'c' aren't co close:
            divides "big" triangle into parallelograms and calcs its centers
        My addition - find the normals!
        '''
        if self.calculate_distance(a, c) / (1.1 * self.step) < 1:
            N = 1
        else:
            N = int(self.calculate_distance(a, c) / self.step) + 1
        As = [a]
        Bs = [b]
        if N > 1:
            for i in range(1, N):
                As.append([a[0] + (c[0] - a[0]) * i / N,
                           a[1] + (c[1] - a[1]) * i / N,
                           a[2] + (c[2] - a[2]) * i / N])
                Bs.append([b[0] + (c[0] - b[0]) * i / N,
                           b[1] + (c[1] - b[1]) * i / N,
                           b[2] + (c[2] - b[2]) * i / N])
        for i in range(0, len(As) - 1):
            self.ias_xyz.append(self.find_central_point([As[i], As[i + 1], Bs[i + 1], Bs[i]]))
            normal = np.cross(np.array(As[i + 1]) - np.array(self.find_central_point([As[i], As[i + 1], Bs[i + 1], Bs[i]])),
                              (As[i]) - np.array(self.find_central_point([As[i], As[i + 1], Bs[i + 1], Bs[i]])))
            self.ias_normals.append(normal / np.linalg.norm(normal))
            self.ias_S.append(self.calculate_square([As[i], As[i + 1], Bs[i + 1], Bs[i]]))
        self.ias_xyz.append(self.find_central_point([As[-1], Bs[-1], c]))
        normal = np.cross(np.array(As[-1]) - np.array(self.find_central_point([As[-1], Bs[-1], c])),
                          np.array(Bs[-1]) - np.array(self.find_central_point([As[-1], Bs[-1], c])))
        self.ias_normals.append(normal / np.linalg.norm(normal))
        self.ias_S.append(self.calculate_square([As[-1], Bs[-1], c]))

    def divide_sector(self, i, j):
        '''
        Divides sector between i-th and j-th paths to small triangles
        '''
        dmin=self.step*0.9
        N = len(self.paths[i])
        M = len(self.paths[j])
        path1 = [self.centre] + self.paths[i]
        path2 = [self.centre] + self.paths[j]
        # first triangle
        self.add2ias(path1[0], path1[1], path2[1])
        # next iterations
        i = 1; j = 1  # now we can redefine i and j for number of point at the path
        while i < N or j < M:
            # check bootlegger reverse
            bootlegger = [False, False]
            if j < M:
                a1 = self.calculate_angle(path1[i], path2[j - 1], path2[j])
                a2 = self.calculate_angle(path1[i], path2[j], path2[j + 1])
                testvec1=np.array(path1[i])-np.array(path1[i-1])
                testvec2=np.array(path2[j])-np.array(path2[j-1])
                angle=np.arccos(np.dot(testvec1,testvec2)/np.linalg.norm(testvec1)/np.linalg.norm(testvec2))/np.pi*180
                if a2 > 170 and a2 < a1:
                    bootlegger[0] = True
                #if angle>160:
                #    print('breaking, angle condition')
                    break
            if i < N:
                a1 = self.calculate_angle(path2[j], path1[i - 1], path1[i])
                a2 = self.calculate_angle(path2[j], path1[i], path1[i + 1])
                if a2 > 170 and a2 < a1:
                    bootlegger[1] = True
            # choose triangle
            if j == M or bootlegger[0]:
                d1 = float('inf')
            else:
                d1 = self.calculate_distance(path1[i], path2[j + 1])
            if i == N or bootlegger[1]:
                d2 = float('inf')
            else:
                d2 = self.calculate_distance(path2[j], path1[i + 1])
            if d1 == d2 == float('inf'):
                break
            if j>20 and self.ias_S[-1]<0.0085*self.step**2:
                print('breaking, triangle area condition')
                break
            elif d1 <= d2:
                self.add2ias(path2[j], path2[j + 1], path1[i])
                j += 1
            else:
                self.add2ias(path1[i], path1[i + 1], path2[j])
                i += 1

    def divide_IAS(self, again=False):
        '''
        Divides IAS to small triangles for the further integration
        '''
        print('Starting IAS division!')
        if self.ias_divided and not again:
            print('IAS was already divided')
            return None
        self.ias_xyz = []
        self.ias_normals = []
        self.ias_S = []
        for i in range(len(self.paths)):
            j = (i + 1) % len(self.paths)
            self.divide_sector(i, j)
        self.ias_divided = True

    def save_IAS(self, folder, name='x', atoms=False):
        '''
        Saves xyz and S
        '''
        if not self.ias_divided:
            raise ('No IAS was generated!')
        with open(os.path.join(folder, '{0}.xyz'.format(name)), 'w') as outf:
            if atoms:
                outf.writelines('\n'.join(
                    [str(len(self.ias_xyz))] + ['0  {0[0]: >13.10f} {0[1]: >13.10f} {0[2]: >13.10f}'.format(_) for _ in
                                                self.ias_xyz]))
            else:
                outf.writelines('\n'.join(
                    [str(len(self.ias_xyz))] + ['{0[0]: >13.10f} {0[1]: >13.10f} {0[2]: >13.10f}'.format(_) for _ in
                                                self.ias_xyz]))
        with open(os.path.join(folder, '{0}.S'.format(name)), 'w') as outf:
            outf.writelines('\n'.join(['{0: >13.10f}'.format(_) for _ in self.ias_S]))
        with open(os.path.join(folder, '{0}.norm'.format(name)), 'w') as outf:
            outf.writelines(
                '\n'.join(['{0[0]: >13.10f}  {0[1]: >13.10f} {0[2]: >13.10f}'.format(_) for _ in self.ias_normals]))


exe_path = os.path.abspath(os.path.dirname(sys.argv[0]))
systeminput = sys.argv
inputfile = systeminput[1]

for i in systeminput:
    if '-sysdiv=' in i:
        DivSysParam = [i[8],i[9:]]
    if '-ifunc=' in i:
        fmode=i[7]
        userf=i[8:]
    if '-imode=' in i:
        imode=i[7:]
    if '-gensurf=' in i:
        gensurf=[i[9],i[10:]]
    if '-isocut=' in i:
        Isosurface_value=float(i[8:])
    if '-useoldsurfs=' in i:
        useoldsurfs=i[-1]

if 'DivSysParam' not in globals():
    DivSysParam=['n','1']
if 'fmode' not in globals() or 'userf' not in globals():
    fmode='x'; userf='1'
if 'imode' not in globals():
    imode='Y'
if 'gensurf' not in globals():
    gensurf=['x','1']
if 'Isosurface_value' not in globals():
    Isosurface_value=0.0
if 'useoldsurfs' not in globals():
    useoldsurfs='n'
    
    
GenSurp=GenSurp(inputfile)
nproc=GenSurp.nproc
if useoldsurfs=='y':
    print('Using already generated surface!')
    GenSurp.DivideSys(DivSysParam=DivSysParam)
    CoreClusterFile='Core-Cluster_BCPs_'+GenSurp.outfilename+'.txt'
elif useoldsurfs=='c':
    print('Re-generating CP list, using new CPs!')
    GenSurp.Generate_BCPlist(DivSysParam, gensurf)
    oldBCP=[]
    newBCP=[]
                        
    with open('Core-Cluster_BCPs.txt') as out:
        for line in out:
            newBCP.append([x for x in line.split()])
            
    if os.path.exists('Core-Cluster_BCPs_'+GenSurp.outfilename+'.txt'):
        with open('Core-Cluster_BCPs_'+GenSurp.outfilename+'.txt') as out:
            for line in out:
                oldBCP.append([x for x in line.split()])
        notnew=[]
        for i,ncp in enumerate(newBCP):
            for ocp in oldBCP:
                if np.linalg.norm(np.array([float(x) for x in ocp[2:]])-np.array([float(x) for x in ncp[2:]]))<0.03:
                    notnew.append(i)
        
        with open('Core-Cluster_BCPs_'+GenSurp.outfilename+'.txt','a') as out:
            for i,ncp in enumerate(newBCP):
                if i not in notnew:
                    out.write('{0[0]} {0[1]} {0[2]} {0[3]} {0[4]}\n'.format(ncp))
    
        with open('Core-Cluster_BCPs.txt','w') as out:
            for i,ncp in enumerate(newBCP):
                if i not in notnew:
                    out.write('{0[0]} {0[1]} {0[2]} {0[3]} {0[4]}\n'.format(ncp))
    
        GenSurp.Generate_Surfaces(DivSysParam='a', isov=Isosurface_value, gensurf=gensurf)
        os.rename('Core-Cluster_BCPs.txt','Core-Cluster_BCPs'+GenSurp.outfilename+'_new.txt')
        CoreClusterFile='Core-Cluster_BCPs_new.txt'
        
    else:
        GenSurp.Generate_Surfaces(DivSysParam='a', isov=Isosurface_value, gensurf=gensurf)
        CoreClusterFile='Core-Cluster_BCPs.txt'
        
    
else:
    GenSurp.Generate_Surfaces(DivSysParam=DivSysParam, isov=Isosurface_value, gensurf=gensurf)
    CoreClusterFile='Core-Cluster_BCPs.txt'
    


with open(CoreClusterFile) as BCPs:
    for nCP,line in enumerate(BCPs):
        CurrentBCP=line.split()
        csurf_name = GenSurp.outfilename + '_surp_' + CurrentBCP[0] + '_' + CurrentBCP[1] + '_f.txt'
        
        if useoldsurfs=='y' and os.path.exists('temp_'+GenSurp.outfilename+'_'+str(nCP)+'.S') and os.path.exists('temp_'+GenSurp.outfilename+'_'+str(nCP)+'.xyz')       and os.path.exists('temp_'+GenSurp.outfilename+'_'+str(nCP)+'.norm'):
            print('Using already generated temp files!')
            temp='temp_'+GenSurp.outfilename+'_'+str(nCP)
        else:    
            y = IAS(csurf_name)
            y.divide_IAS()
            y.save_IAS(exe_path, 'temp')
            temp='temp'
        
        CurrentBCPcoord=np.array([float(coord) for coord in CurrentBCP[2:]])

        for nAt, Atom in enumerate(GenSurp.CENTERS):
            if CurrentBCP[0] in Atom:
                iaxyz = np.array(Atom[1:-1])
                iacharge = float(Atom[0])
            if CurrentBCP[1] in Atom:
                oaxyz = np.array(Atom[1:-1])
                oacharge = float(Atom[0])
        print('Surface for BCP', CurrentBCP[:2],  'generated!')

        R_intermolecular = np.array([p-q for p,q in zip(oaxyz,iaxyz)])

        # define normals coords from .norm file
        normals = []
        with open(temp+'.norm', 'r') as inpf:
            text = [_.strip() for _ in inpf.readlines()]
        for line in text:
            normals.append([float(_) for _ in line.split()])
        normals=np.array(normals)

        # define triangulated points' coords from .xyz file excluding their quantity
        xyz = []
        with open(temp+'.xyz', 'r') as inpf:
            text = [_.strip() for _ in inpf.readlines()]
        for line in text:
            if len(line.split()) == 3:
                xyz.append([float(_) for _ in line.split()])
        xyz=np.array(xyz)

        if temp=='temp':
            # REVERSE normals - to make them point at the outer atom!!!
            for s in range(len(normals)):
                if np.linalg.norm(xyz[s]-CurrentBCPcoord)<0.09:
                    if np.linalg.norm(xyz[s] + normals[s] - np.array(iaxyz)) < np.linalg.norm(xyz[s] - normals[s] - np.array(iaxyz)):
                        normals[s] = normals[s] * (-1)
                else:
                    try:
                        prev_norm = xyz[s-1] + normals[s-1]
                        if np.linalg.norm(xyz[s] + normals[s] - prev_norm) > np.linalg.norm(xyz[s] - normals[s] - prev_norm):
                            normals[s] = normals[s] * (-1)
                    except IndexError:
                        pass
                
            with open('temp.norm', 'w') as inpf:
                for norm in normals:
                    inpf.write('{0} {1} {2}\n'.format(norm[0],norm[1],norm[2]))


        # define Rs (distance from atom to point on surface)
        Rs = []
        if imode == 'Y+R':
            for a in xyz:
                Rs.append([p-q for p,q in zip(a,iaxyz)])
        elif imode == 'Y+Re' or imode == 'RedY':
            for Atom in GenSurp.CENTERS:
                Rss = []
                for a in xyz:
                    Rss.append([p-q for p,q in zip(a,Atom[1:-1])])
                Rs.append(Rss)
        elif imode in ['Dipole','N_v2','Y_v2','Y_v3','DI','MSeries','101','102']:
            for a in xyz:
                rs1=np.array([p-q for p,q in zip(a,iaxyz)])
                rs2=np.array([p-q for p,q in zip(a,oaxyz)])
                Rs.append([rs1,rs2])

            

        # run Multiwfn (or don't run)
        print("Calculating surface integral")
        if imode not in ['S','MSeries']:
            if fmode == 'u':
                text = ['1000', '10', nproc, '1000', '2', userf, '5', '100', '100', temp+'.xyz', 'temp.prop']
            else:
                text = ['1000', '10', nproc, '5', userf, '100', temp+'.xyz', 'temp.prop']
            with open('temp.wfn', 'w') as outf:
                outf.write('\n'.join(text) + '\n')
            os.system('{0} {1} < {2} &> /dev/null'.format('./MultiwfnTest', inputfile, 'temp.wfn'))

            if userf not in ["-1488","-228"]:
                # read properties and surfaces for integration
                with open(os.path.join(exe_path, 'temp.prop'), 'r') as inpf:
                    X = [float(_.strip().split()[-1]) for _ in inpf.readlines()[1:]]
                                
            elif userf == "-1488":
                # read properties and surfaces for integration
                with open(os.path.join(exe_path, 'temp.prop'), 'r') as inpf:
                    X=[]
                    for _ in inpf.readlines()[1:]: 
                       X.append(np.array([float(x) for x in _.strip().split()[3:]])) 

            elif userf == "-228":
                # read properties and surfaces for integration
                with open(os.path.join(exe_path, 'temp.prop'), 'r') as inpf:
                    X=[]
                    for _ in inpf.readlines()[1:]: 
                       _=[float(x) for x in _.strip().split()[3:]]
                       X.append(np.array([_[0:3],[_[1]]+_[3:5],[_[2],_[4],_[5]]])) 

        print('X[0] is ',X[0])

        with open(os.path.join(exe_path, temp+'.S'), 'r') as inpf:
            S = [float(_.strip()) for _ in inpf.readlines()]

        # output
        if imode == 'S':
            integration_result = sum(s for s in S)
        
        elif imode == 'N' and userf not in ["-228","-1488"]:
            integration_result = sum([x * s for x, s in zip(X, S)])
        
        elif imode == 'Y' and userf not in ["-228","-1488"]:
            integration_result = sum([x * s * np.dot(np.array(n), R_intermolecular) for x, s, n in zip(X, S, normals)])
        
        elif imode == 'Y+R':
            integration_result = sum([x * s * np.dot(np.array(n), np.array(R_at_surf)) for x, s, n, R_at_surf in zip(X, S, normals, Rs)])
        
        elif imode == 'Y+Re':
            integration_result = []
            for nRs, Atom in enumerate(GenSurp.CENTERS):
                R_intermolecular = np.array([p-q for p,q in zip(Atom[1:-1], iaxyz)])
                if np.linalg.norm(R_intermolecular)!=0:
                    result = sum([x * s * np.dot(np.array(n)*(-1), np.array(R_at_surf)) for x, s, n, R_at_surf in
                              zip(X, S, normals, Rs[nRs])])
                else:
                    result = sum([x * s * np.dot(np.array(n), np.array(R_at_surf)) for x, s, n, R_at_surf in
                                  zip(X, S, normals, Rs[nRs])])
                integration_result.append([Atom[4], result, np.linalg.norm(R_intermolecular)])
        
        elif imode == 'N_v2':
            ir1=0
            ir2=0
            for x, s, rss in zip(X, S, Rs): 
                ir1 += x*s/np.linalg.norm(rss[0])
                ir2 += x*s/np.linalg.norm(rss[1])
            integration_result=[ir1,ir2,np.linalg.norm(R_intermolecular)]

        elif imode == 'Y_v2':
            ir1=0
            ir2=0
            for x, s, n, rss in zip(X, S, normals,Rs):
                ir1 += x*s*np.dot(np.array(n), np.array(rss[0]))/np.linalg.norm(rss[0])
                ir2 += x*s*np.dot(np.array(n)*(-1), np.array(rss[1]))/np.linalg.norm(rss[1])
            integration_result=[ir1,ir2,np.linalg.norm(R_intermolecular)]

        elif imode == 'Y' and userf =="-228":
            integration_result = -sum([s * np.dot(np.array(n), x) for x, s, n, in zip(X, S, normals)])
            print("integration_result", integration_result)

        elif imode == 'N' and userf == "-228":
            integration_result = sum([np.trace(x) * s for x, s in zip(X, S)])
            print("integration_result", integration_result)

        surface_area = sum(s for s in S)

        # remove temporary files
        #for ff in ['S', 'prop', 'wfn', 'xyz', 'norm']:
        #    try:
        #        os.remove(os.path.join(exe_path, 'temp.{0}'.format(ff)))
        #    except FileNotFoundError:
        #        pass

        # rename temporary files
        for ff in ['S', 'prop', 'wfn', 'xyz', 'norm']:
            try:
                shutil.move('temp.{0}'.format(ff),'temp_'+GenSurp.outfilename+'_'+str(nCP)+'.{0}'.format(ff))
            except FileNotFoundError:
                pass
                                  
        currdir = os.getcwd()
        maindir = currdir + '/' + GenSurp.outfilename+'.'
        if os.path.exists(maindir) == False:
            os.mkdir(maindir)

        currmode = imode + fmode + userf + gensurf[0]+gensurf[1]

        if os.path.exists(maindir + '/' + currmode) == False:
            os.mkdir(maindir + '/' + currmode)
        OUT_name = maindir + '/' + currmode + '/IntegrationResult.txt'

        if imode == 'Y+Re':
            with open(OUT_name, 'a') as Energy_DATA:
                for result in integration_result:
                    Energy_DATA.write(str(CurrentBCP[0])), Energy_DATA.write('   '), Energy_DATA.write(
                        str(CurrentBCP[1]))
                    Energy_DATA.write('\t'), Energy_DATA.write(str(result[1]))
                    Energy_DATA.write('\tCurrent outer atom: '), Energy_DATA.write(result[0])
                    Energy_DATA.write('\tDist to current outer atom: '), Energy_DATA.write(str(result[2]))
                    Energy_DATA.write('\tSurface area: '), Energy_DATA.write(str(surface_area))
                    Energy_DATA.write('\n')

        elif imode == 'N_v2':
            with open(OUT_name, 'a') as Energy_DATA:
                Energy_DATA.write(str(CurrentBCP[0])), Energy_DATA.write('   '), Energy_DATA.write(str(CurrentBCP[1]))
                Energy_DATA.write('\t')
                ir=integration_result
                Energy_DATA.write('{0} {1} {2}'.format(ir[0]+ir[1],(ir[0]+ir[1])/ir[2],ir[2]))
                Energy_DATA.write("\t{0}\t".format(surface_area))
                Energy_DATA.write('{0} {1}\n'.format(ir[0],ir[1]))

        elif imode=='Y_v2':
            with open(OUT_name, 'a') as Energy_DATA:
                Energy_DATA.write(str(CurrentBCP[0])), Energy_DATA.write('   '), Energy_DATA.write(str(CurrentBCP[1]))
                Energy_DATA.write('\t')
                ir=integration_result
                Energy_DATA.write('{0} {1} {2}'.format(ir[0]+ir[1],(ir[0]+ir[1])/ir[2],ir[2]))
                Energy_DATA.write("\t{0}\t".format(surface_area))
                Energy_DATA.write('{0} {1}\n'.format(ir[0],ir[1]))

        elif imode == 'Y' and userf == "-228":
            with open(OUT_name, 'a') as Energy_DATA:
               Energy_DATA.write('{0}\t{1}\t'.format(CurrentBCP[0],CurrentBCP[1]))
               Energy_DATA.write(f'{integration_result[0]}\t{integration_result[1]}\t{integration_result[2]}\n')
                                             
                                
        else:
            with open(OUT_name, 'a') as Energy_DATA:
                Energy_DATA.write('{0}\t{1}\t'.format(CurrentBCP[0],CurrentBCP[1]))
                Energy_DATA.write('{0}\t'.format(integration_result))
                Energy_DATA.write('{0}\t'.format(integration_result / np.linalg.norm(R_intermolecular)))
                Energy_DATA.write('{0}\t'.format(np.linalg.norm(R_intermolecular)))
                Energy_DATA.write("\t{0}\n".format(surface_area))


    ################################## del ##################################

    for ff in ['.wfn', '.txt', '2.wfn', '2.txt', '3.wfn', '3.txt']:
        try:
            os.remove('myprog{0}'.format(ff))
        except FileNotFoundError:
            pass
        
try:
    shutil.move('Core-Cluster_BCPs.txt', 'Core-Cluster_BCPs_'+GenSurp.outfilename+'.txt')
except FileNotFoundError:
    pass
