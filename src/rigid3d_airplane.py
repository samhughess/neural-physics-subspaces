import jax.numpy as jnp
from jax import grad, jit, vmap, device_get
#import fixed_point_projection
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as psim
import aerosandbox as asb
import aerosandbox.numpy as asbnp
import scipy.spatial.transform as transform
import jax,ll2,op2

#import jax_transformations3d as jaxtran

try:
    import igl
finally:
    print("WARNING: igl bindings not available")

import utils
import config

## Define functions

def vortexlatticemethod(airplane, q, wind):
    
    qR = q.reshape(-1,12,1)
    vel_array = jnp.array([qR[-1, 1, 1], qR[-1, 3, 1], qR[-1, 5, 1]])
    vel_relative = jnp.add(vel_array, wind)
    velocity = jnp.linalg.norm(vel_relative) # m/s
    #alpha should be arctanh but not working correctly
    alpha = jnp.arctanh(jnp.true_divide(vel_relative[5], vel_relative[1])) # deg
    vlm = asb.VortexLatticeMethod(airplane=airplane, 
                                  op_point=asb.OperatingPoint(
                                        velocity = velocity, # in m/s
                                        alpha = alpha, # in degrees
                                  )
    )
    aero = vlm.run()
    return aero # Note: aero is a dictionary object with L, D, Y, l, m ,n, CL, CD, CU, Cl, CD, etc... F_g, F_w, M_g, M_w



def liftinglinemethod(airplane, q, wind):
  
    qR = q.reshape(-1,12,1)
    vel_array = jnp.array([qR[-1, 1, 1], qR[-1, 3, 1], qR[-1, 5, 1]])
    vel_relative = jnp.add(vel_array, wind)
    velocity = jnp.linalg.norm(vel_relative) # m/s
    #these were originally arctanh but had to change because wouldn't run
    alpha = jnp.arctan(jnp.true_divide(vel_relative[5], vel_relative[1])) # deg
    beta = jnp.arctan(jnp.true_divide(vel_relative[3], vel_relative[1])) # deg
    p = qR[-1,7,1] # rad/s
    q = qR[-1,9,1] # rad/s
    r = qR[-1,11,1] # rad/s
    op_point = op2.OperatingPoint2(p,q,r,velocity, alpha, beta)
    analysis = ll2.LiftingLine(airplane,op_point)
    analysis.run()
    return analysis



def make_body(file, scale, density, inertia):
    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    # c is the initial center of mass
    c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:,None]*W) * density
    # TODO: Translate to use quaternion rather than euler angles to avoid gimbal lock

    # x0 is the initial state space with x, x_dot, y, y_dot, z, z_dot, yaw, yaw_dot (p), pitch, pitch_dot (q), roll, roll_dot (r) 
    # x0 = jnp.array([cg, omega])
    x0 = jnp.array([c[0], 0, c[1], 0, c[2], 0, 0, 0, 0, 0, 0, 0])


    body = {'v': v, 'f': f, 'W': W, 'x0': x0, 'mass': mass, 'inertia': inertia}
    return body

def make_body_noinput(file, density, scale):
    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    # c is the initial center of mass
    c = jnp.sum( vol[:,None]*v, axis=0 ) / jnp.sum(vol) 
    v = v - c
    W = jnp.c_[v, jnp.ones(v.shape[0])]
    mass = jnp.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [[1, 0, 0],[0, 1, 0],[0, 0, 1], c] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': mass }
    return body

def make_joint(b0, b1, bodies, joint_pos_world, joint_vec_world ):
    # Creates a joint between the specified bodies, assumes the bodies have zero rotation and are properly aligned in the world
    # TODO: Use rotation for joint initialization
    pb0 = joint_pos_world
    vb0 = joint_vec_world
    if b0 != -1:
        c0 = bodies[b0]['x0'][3,:]
        pb0 = pb0 - c0
    pb1 = joint_pos_world
    vb1 = joint_vec_world
    if b1 != -1:
        c1 = bodies[b1]['x0'][3,:]
        pb1 = pb1 - c1
    joint = {'body_id0': b0, 'body_id1': b1, 'pos_body0': pb0, 'pos_body1': pb1, 'vec_body0': vb0, 'vec_body1': vb1}
    return joint

def bodiesToStructOfArrays(bodies):
    v_arr = []
    f_arr = []
    W_arr = []
    x0_arr = []
    mass_arr = []
    inertia_arr = []

    for b in bodies:
        v_arr.append(b['v'])
        f_arr.append(b['f'])
        W_arr.append(b['W'])
        x0_arr.append(b['x0'])
        mass_arr.append(b['mass'])
        inertia_arr.append(b['inertia'])
    
    out_struct = {
        'v'     : jnp.stack(v_arr, axis=0),
        'f'     : jnp.stack(f_arr, axis=0),
        'W'     : jnp.stack(W_arr, axis=0),
        'x0'    : jnp.stack(x0_arr, axis=0),
        'mass'  : jnp.stack(mass_arr, axis=0),
        'inertia' : jnp.stack(inertia_arr, axis=0),
    }

    n_bodies = len(v_arr)

    return out_struct, n_bodies


class Aircraft:
    
    def update(self, system_def):
        self.mass = system_def['mass']
        self.gravity = system_def['gravity']
        self.inertia = system_def['inertia']
        self.ext_windx = system_def['external_forces']['wind_strength_x']
        self.ext_windy = system_def['external_forces']['wind_strength_y']
        self.ext_windz = system_def['external_forces']['wind_strength_z']
        self.ext_thrust_left = system_def['external_forces']['thrust_strength_left']
        self.ext_thrust_right =  system_def['external_forces']['thrust_strength_right']
    
    @staticmethod
    def construct(problem_name):
        system_def = {}
        system = Aircraft()
        # Chosen default parameters:
        system.system_name = "Aircraft"
        system.problem_name = str(problem_name)
        system_def['cond_param'] = jnp.zeros((0,)) # a length-0 array
        system_def['external_forces'] = {}
        system_def["contact_stiffness"] = 1000000.0
        system_def['airplane'] = {}
        system.cond_dim = 0
        system.body_ID = None

        bodies = []
        joint_list = []
        numBodiesFixed = 0

        if problem_name == 'generic_airplane':
            scale = 1

            #why is this a 3x3 rather than a 4x4
            mass = 3*jnp.eye(3)
            inertia = jnp.eye(3)
            
            # Add all the necessary bodies
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), scale, 1, inertia))

            numBodiesFixed = 0

            # Define the external forces
            system_def["gravity"] = jnp.array([0.0, 0.0, -9.8])

            system_def['external_forces']['aero_force'] = 0

            system_def['external_forces']['wind_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['wind_strength_x'] = 0.0
            system_def['external_forces']['wind_strength_y'] = 0.0
            system_def['external_forces']['wind_strength_z'] = 0.0

            system_def['external_forces']['thrust_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['thrust_strength_left'] = 0.0
            system_def['external_forces']['thrust_strength_right'] = 0.0

            system.body_ID = np.array([1])

            # Define airplane parameters for aero calcs
            name_airplane = "generic_testplane"
            airfoil_airplane = "naca0012"
            symmetry = True

            # Declare the airplane
            wing_airfoil = asb.Airfoil(airfoil_airplane)
            airplane = asb.Airplane(name = name_airplane,
                                    xyz_ref = [0, 0, 0], # Cg location
                                    wings=[
                                        asb.Wing(
                                            name = "Main Wing",
                                            symmetric = True,
                                            xsecs = [
                                                asb.WingXSec(
                                                    xyz_le = [0, 0, 0],
                                                    chord = 0.16,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.01, 0.5, 0],
                                                    chord = 0.08,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.8, 1, 0.1],
                                                    chord = 0.04,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                            ]
                                        ),
                                    ],
                                    fuselages = [
                                        asb.Fuselage(
                                            name="Fuselage",
                                            xsecs = [
                                                asb.FuselageXSec(
                                                    xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                                                    radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
                                                )
                                                for xi in asbnp.cosspace(0, 1, 30)
                                            ]
                                        )
                                    ]
                                )
            
            system.airplane = airplane

            
        elif problem_name == 'stoprotor_vtol':
            # TODO: add airplane definition
            scale = 1
            
            # Add all the necessary bodies
            # Main Body
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Left Counterbalance
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Right Counterbalance
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Top Rotor
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Right Wing
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Left Wing
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            
            system.body_ID = np.array([0, 1, 2, 3, 4, 5])

            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

            system_def["gravity"] = jnp.array([0.0, 0.0, -9.8])

            system_def['external_forces']['aero_force'] = 0
            
            system_def['external_forces']['wind_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['wind_strength_x'] = 0.0
            system_def['external_forces']['wind_strength_y'] = 0.0
            system_def['external_forces']['wind_strength_z'] = 0.0

            system_def['external_forces']['thrust_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['thrust_strength_left'] = 0.0
            system_def['external_forces']['thrust_strength_right'] = 0.0

            system_def['external_forces']['thrust_angle_minmax'] = (-90, 90) # in m/s
            system_def['external_forces']['thrust_angle_left'] = 0.0
            system_def['external_forces']['thrust_angle_right'] = 0.0

            system_def['external_forces']['toprotor_velocity_minmax'] = (-90, 90) # in m/s
            system_def['external_forces']['toprotor_velocity'] = 0.0

            # Define airplane parameters for aero calcs
            name_airplane = "stoprotor_vtol"
            airfoil_airplane = "clarky"
            symmetry = False

            # Declare the airplane
            wing_airfoil = asb.Airfoil(airfoil_airplane)
            airplane = asb.Airplane(name = name_airplane,
                                    xyz_ref = [0, 0, 0], # Cg location
                                    wings=[
                                        asb.Wing(
                                            name = "Right Wing",
                                            symmetric = False,
                                            xsecs = [
                                                asb.WingXSec(
                                                    xyz_le = [0.04, 0.05, -0.085],
                                                    chord = -0.16,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.018, 0.170, -0.085],
                                                    chord = -0.112,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [-0.0025, 0.3, -0.085],
                                                    chord = -0.064,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                            ]
                                        ),
                                        asb.Wing(
                                            name = "Left Wing",
                                            symmetric = False,
                                            xsecs = [
                                                asb.WingXSec(
                                                    xyz_le = [-0.04, -0.05, -0.085],
                                                    chord = 0.16,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [-0.018, -0.170, -0.085],
                                                    chord = 0.112,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.0025, -0.3, -0.085],
                                                    chord = 0.064,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                            ]
                                        ),
                                    ],
                                    fuselages = [
                                        asb.Fuselage(
                                            name="Fuselage",
                                            xsecs = [
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.1,
                                                    height = 0.24,
                                                    shape = 100, 
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                            ]
                                        )
                                    ]
                                )
            
            system.airplane = airplane


            
        elif problem_name == 'stoprotor_forwardflight':
            # TODO: add airplane definition
            scale = 1
            
            # Add all the necessary bodies
            # Main Body
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Left Counterbalance
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Right Counterbalance
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Top Rotor
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Right Wing
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
            # Left Wing
            bodies.append( make_body( os.path.join(".", "data", "Body1.obj"), 1000, scale))
             
            system.body_ID = np.array([0, 1, 2, 3, 4, 5])
           
            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

            system_def["gravity"] = jnp.array([0.0, 0.0, -9.8])

            system_def['external_forces']['aero_force'] = 0
            
            system_def['external_forces']['wind_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['wind_strength_x'] = 0.0
            system_def['external_forces']['wind_strength_y'] = 0.0
            system_def['external_forces']['wind_strength_z'] = 0.0

            system_def['external_forces']['thrust_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['thrust_strength_left'] = 0.0
            system_def['external_forces']['thrust_strength_right'] = 0.0

            system_def['external_forces']['thrust_angle_minmax'] = (-90, 90) # in m/s
            system_def['external_forces']['thrust_angle_left'] = 0.0
            system_def['external_forces']['thrust_angle_right'] = 0.0

            # Define airplane parameters for aero calcs
            name_airplane = "stoprotor_vtol"
            airfoil_airplane = "clarky"
            symmetry = True

            # Declare the airplane
            wing_airfoil = asb.Airfoil(airfoil_airplane)
            airplane = asb.Airplane(name = name_airplane,
                                    xyz_ref = [0, 0, 0], # Cg location
                                    wings=[
                                        asb.Wing(
                                            name = "Right Wing",
                                            symmetric = True,
                                            xsecs = [
                                                asb.WingXSec(
                                                    xyz_le = [0.04, 0.05, -0.085],
                                                    chord = 0.16,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.018, 0.170, -0.085],
                                                    chord = 0.112,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [-0.0025, 0.3, -0.085],
                                                    chord = 0.064,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                            ]
                                        ),
                                    ],
                                    fuselages = [
                                        asb.Fuselage(
                                            name="Fuselage",
                                            xsecs = [
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.1,
                                                    height = 0.24,
                                                    shape = 100, 
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                            ]
                                        )
                                    ]
            )

        else:
            raise ValueError("could not parse problem name: " + str(problem_name))

        posFixed  = jnp.array( np.array([ body['x0']   for body in bodies[0:numBodiesFixed] ]).flatten() )
        pos  = jnp.array( np.array([ body['x0']   for body in bodies[numBodiesFixed:] ]).flatten() )

        mass = jnp.array( np.array([ body['mass'] for body in bodies[numBodiesFixed:] ]).flatten() )
        inertia = jnp.array( np.array([ body['inertia'] for body in bodies[numBodiesFixed:] ]).flatten())
        
        #
        system.dim = pos.size

        system.bodiesRen = bodies
        system.n_bodies = len(bodies)
        
        #
        system.joints = joint_list

        system_def['fixed_pos'] = posFixed
        system_def['rest_pos'] = pos
        system_def['init_pos'] = pos
       
        system_def['mass'] = mass
     

        system_def['inertia'] = inertia

        system_def['interesting_states'] = system_def['init_pos'][None,:]

        #val = jnp.sum(jnp.square(pos), axis = 0) part of potential inertia calc

        return system, system_def
  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system


        
    def potential_energy(self, system_def, q):
  
        #used the following line when not simulating
        #self.update(system_def)
        qR = q.reshape(-1,12,1)
        #q_pos = qR[::2]
        #q_xyz = q_pos[0:2]
        
        #q_abs = jnp.absolute(qR)
        q_nonneg = jax.tree_map(lambda x: jnp.maximum(x, 0), qR)
        # Create an array of even indices
        indices_even = jnp.arange(0, len(qR[0]), 2)

        # Use np.take to extract elements at even indices
        q_pos = jnp.take(q_nonneg, indices_even)
        q_xyz1 = jnp.take(q_pos, np.array([0, 1, 2]))
        q_xyz = jnp.append(q_xyz1, 1)
        
        #mass = system_def['mass']
        massR = self.mass.reshape(-1,4,4)
        gravity = self.gravity
    
        gravity_app = jnp.append(gravity, 1)
        c_weighted = massR*q_xyz
        
        gravity_energy = -jnp.sum(c_weighted * gravity_app[None,:])
      
        return gravity_energy
        
    def ke_error(self, system_def, q, q_dot):
        #changed dimensions here
        #q_dotR = q_dot.reshape(-1,12,1)
        #changed this to get working so check
        q_dotR = q_dot.reshape(-1,4,3)
        #legit just killing the last dimension here not sure if great idea
        q_new = q_dotR[:3]
        #is this supposed to be a 4x4??
        #massR = system_def['mass'].reshape(-1,4,4)
        massR = self.mass.reshape(-1,4,4)
        
        A = jnp.swapaxes(q_dotR,1,2)@ massR @ q_new
        
        Ke_offset = 0.5*jnp.sum(jnp.trace(A, axis1=1, axis2=2))
     
        return Ke_offset
    
    
    def action(self, system, system_def, q):
        # TODO add all forces
        PE = self.potential_energy(system_def, q)
        ke_transl = self.ke_translation(system_def, q)
        ke_rot = self.ke_rotational(system_def, q)
        KE = ke_transl + ke_rot       
        print('test5', jnp.shape(ke_transl), jnp.shape(ke_rot))
        lagrangian = KE+PE
        n_aero = 2
        n_thrust = 0
        #windx = system_def['external_forces']['wind_strength_x']
        #windy = system_def['external_forces']['wind_strength_y']
        #windz = system_def['external_forces']['wind_strength_z']
        windx = self.ext_windx
        windy = self.ext_windy
        windz = self.ext_windz
        wind = jnp.array([windx, windy, windz])
     
        #need to figure out what looking for here since lifting line has no return value

        aero_data = liftinglinemethod(self.airplane, q, wind)
    
        # aero_transforce = aero_data['F_b']
        # aero_rotmoment = aero_data['M_b']
       
        aero_data._calculate_vortex_strengths()
        aero_data._calculate_forces()
        
        aero_transforce = aero_data.force_total_inviscid_geometry  
        aero_rotmoment = aero_data.moment_total_inviscid_geometry

        #not exactly sure right values here
        
        #thrust_force_left = system_def['external_forces']['thrust_strength_left']*jnp.array([-1, 0, 0])
        #thrust_force_right =  system_def['external_forces']['thrust_strength_right']*jnp.array([-1, 0, 0])
        
        thrust_force_left = self.ext_thrust_left*jnp.array([-1, 0, 0])
        thrust_force_right =  self.ext_thrust_right*jnp.array([-1, 0, 0])
        
        #changed from system to self
        dis_aero_translation = self.dissipation_fnc(n_aero, aero_transforce, q, "translation")
        dis_aero_rotation = self.dissipation_fnc(n_aero, aero_rotmoment, q, "rotation")
        dis_thrust_left = self.dissipation_fnc(n_thrust, thrust_force_left, q, "translation")
        dis_thrust_right = self.dissipation_fnc(n_thrust, thrust_force_right, q, "translation")

        dissipation = dis_aero_translation + dis_aero_rotation + dis_thrust_left + dis_thrust_right
       
        print('test11', jnp.shape(lagrangian), jnp.shape(dissipation))
        return lagrangian + dissipation
       
        

    def ke_translation(self, system_def, q):
        qR = q.reshape(-1,12,1)
        velocity = jnp.array([qR[-1,1,1], qR[-1,3,1], qR[-1,5,1]])
        #velocity_reshaped = jnp.expand_dims(velocity, axis=(0, 2))
        velocity = jnp.append(velocity, 1)
        mass = system_def["mass"]
        massR = mass.reshape(-1,4,4)
        ke_translational = 0.5* jnp.dot(jnp.matmul(jnp.transpose(velocity),massR),velocity)
        return ke_translational[0]
    
    def ke_rotational(self, system_def, q):
        qR = q.reshape(-1,12,1)
        omega = jnp.array([qR[-1,7,1], qR[-1,9,1], qR[-1,11,1]])
        inertia = system_def['inertia']
        inertiaR = inertia.reshape(-1, 3, 3)
        ke_rot = 0.5 * jnp.matmul(jnp.matmul(jnp.transpose(omega),inertiaR),omega)
        return ke_rot[0]
    
    def dissipation_fnc(self, n, c, q, style):
        qR = q.reshape(-1,12,1)
        if style == "translation":
            q_dot = jnp.array([qR[-1,1,1], qR[-1,3,1], qR[-1,5,1]])
        elif style == "rotation":
            q_dot = jnp.array([qR[-1,7,1], qR[-1,9,1], qR[-1,11,1]])
        else:
            print("Specified style not found. Use either ''translation'' or ''rotation'' as style name")

        D = 1/(n+1)*jnp.dot(c,q_dot)
        
        return D


    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        # Sample a random, valid setting of the conditional parameters for the system.
        # (If there are no conditional parameters, returning the empty array as below is fine)
        # TODO implement
        return jnp.zeros((0,))
    
    def build_system_ui(self, system_def):
        if psim.TreeNode("system UI"):
            psim.TextUnformatted("External forces:")

            if "wind_strength_x" in system_def["external_forces"]:
                low, high = system_def['external_forces']['wind_strength_minmax']
                _, new_val = psim.SliderFloat("wind_strength_x", float(system_def['external_forces'][ 'wind_strength_x']), low, high)
                system_def['external_forces']['wind_strength_x'] = jnp.array(new_val)

            if "wind_strength_y" in system_def["external_forces"]:
                low, high = system_def['external_forces']['wind_strength_minmax']
                _, new_val = psim.SliderFloat("wind_strength_y", float(system_def['external_forces'][ 'wind_strength_y']), low, high)
                system_def['external_forces']['wind_strength_y'] = jnp.array(new_val)

            if "wind_strength_z" in system_def["external_forces"]:
                low, high = system_def['external_forces']['wind_strength_minmax']
                _, new_val = psim.SliderFloat("wind_strength_z", float(system_def['external_forces'][ 'wind_strength_z']), low, high)
                system_def['external_forces']['wind_strength_z'] = jnp.array(new_val)


            psim.TreePop()
    
    def align_vectors_wrapper(self, up_vector, face_normal):
        return transform.Rotation.align_vectors(up_vector, face_normal)

    # ===========================================
    # === Visualization routines
    # ===========================================

    def visualize(self, system_def, x, name="rigid3d", prefix='', transparency=1.):

        xr = jnp.concatenate((system_def['fixed_pos'],x)).reshape(-1,4,3)

        for bid in range(self.n_bodies):
            v = np.array(jnp.matmul(self.bodiesRen[bid]['W'], xr[bid]))
            f = np.array(self.bodiesRen[bid]['f'])

            ps_body = ps.register_surface_mesh("body" + prefix + str(bid), v, f)
            if transparency < 1.:
                ps_body.set_transparency(transparency)

            transform = np.identity(4)
            ps_body.set_transform( transform )
        
        return ps_body # not clear that anything needs to be returned
    

    def visualize_set_nice_view(system_def, q):
        # Set a Polyscope camera view which nicely looks at the scene
        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))
        # Example:
        # (could also be dependent on x)
        # ps.look_at((2., 1., 2.), (0., 0., 0.))
    
        pass
