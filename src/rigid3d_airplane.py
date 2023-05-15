import jax.numpy as jnp
from jax import grad, jit, vmap
import fixed_point_projection
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as psim

try:
    import igl
finally:
    print("WARNING: igl bindings not available")

import utils
import config

## Define functions
def make_body(file, density, scale):

    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [[1, 0, 0],[0, 1, 0],[0, 0, 1], c] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': mass }
    return body

def make_joint( b0, b1, bodies, joint_pos_world, joint_vec_world ):
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
    for b in bodies:
        v_arr.append(b['v'])
        f_arr.append(b['f'])
        W_arr.append(b['W'])
        x0_arr.append(b['x0'])
        mass_arr.append(b['mass'])
    
    out_struct = {
        'v'     : jnp.stack(v_arr, axis=0),
        'f'     : jnp.stack(f_arr, axis=0),
        'W'     : jnp.stack(W_arr, axis=0),
        'x0'    : jnp.stack(x0_arr, axis=0),
        'mass'  : jnp.stack(mass_arr, axis=0),
    }

    n_bodies = len(v_arr)

    return out_struct, n_bodies

class Aircraft:

    @staticmethod
    def construct(problem_name):

        '''

        Basic philosophy:
            We define the system via two objects, a object instance ('system'), which can
            hold pretty much anything (strings, function pointers, etc), and a dictionary 
            ('system_def') which holds only jnp.array objects.

            The reason for this split is the JAX JIT system. Data stored in the `system`
            is fixed after construction, so it can be anything. Data stored in `system_def`
            can potentially be modified after the system is constructed, so it must consist
            only of JAX arrays to make JAX's JIT engine happy.


        The fields which MUST be populated are:
       
            = System name
            system.system_name --> str
            The name of the system (e.g. "neohookean")
            

            = Problem name
            system.problem_name --> str
            The name of the problem (e.g. "trussbar2")
       

            = Dimension
            system.dim --> int
                The dimension of the configuration space for the system. When we
                learn subspaces that map from `R^d --> R^n`, this is `n`. If the 
                system internally expands the configuration space e.g. to append 
                additional pinned vertex positions, those should NOT be counted 
                here.


            = Initial position
            system_def['init_pos'] --> jnp.array float dimension: (n,)
                An initial position, used to set the initial state in the GUI
                It does not necessarily need to be the rest pose of the system.
            

            = Conditional parameters dimension
            system.cond_dim --> int
                The dimension of the conditional parameter space for the system. 
                If there are no conditional parameters, use 0.
            
            = Conditional parameters value
            system_def['cond_param'] --> jnp.array float dimension: (c,)
                A vector of conditional paramters for the system, defining its current
                state. If there are no conditional parameters, use a length-0 vector.
            

            = External forces
            system_def['external_forces'] --> dictionary of anything
                Data defining external forces which can be adjusted at runtime
                The meaning of this dictionary is totally system-dependent. If unused, 
                leave as an empty dictionary. Fill the dictionary with arrays or other 
                data needed to evaluate external forces.


            = Interesting states
            system_def['interesting_states'] --> jnp.array float dimension (i,n)
                A collection of `i` configuration state vectors which we want to explicitly track
                and preserve.
                If there are no interesting states (i=0), then this should just be a size (0,n) array.


            The dictionary may also be populated with any other user-values which are useful in 
            defining the system.

            NOTE: When you add a new system class, also add it to the registry in config.py 
        '''

        system_def = {}


        # Chosen default parameters:
        system_def['system_name'] = "aircraft"
        system_def['problem_name'] = str(problem_name)
        system_def['cond_params'] = jnp.zeros((0,)) # a length-0 array
        system_def['external_forces'] = {}

        bodies = []
        joint_list = []
        numBodiesFixed = 0

        if problem_name == 'generic airplane':
            # Add all the necessary bodies
            bodies.append( make_body( os.path.join(".", "data", ".obj")))
            
            # Add all the necessary joints
            joint_list.append( make_joint(0, -1, bodies, jnp.array([ 0,           0.08      ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ))

            # Define the external forces
            system_def["gravity"] = jnp.array([0.0, -0.98, 0.0])
            system_def['external_forces']['']

            config_dim = 280
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))
        
        elif problem_name == 'stoprotor':

            # and so on....
            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

        else:
            raise ValueError("could not parse problem name: " + str(problem_name))


        return system_def
  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system

    def potential_energy(system_def, q):
        # TODO implement
        return 0.
   

    def kinetic_energy(system_def, q, q_dot):
        # TODO implement
        return 0.

    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(system_def, rngkey):
        # Sample a random, valid setting of the conditional parameters for the system.
        # (If there are no conditional parameters, returning the empty array as below is fine)
        # TODO implement
        return jnp.zeros((0,))

    # ===========================================
    # === Visualization routines
    # ===========================================
    
    def build_system_ui(system_def):
        # Construct a Polyscope gui to tweak parameters of the system
        # Make psim.InputFloat etc calls here

        # If appliciable, the cond_params values and external_forces values should be 
        # made editable in this function.

        pass

    def visualize(system_def, q):
        # Create and/or update a Polyscope visualization of the system in its current state

        # TODO implement
        pass
    

    def visualize_set_nice_view(system_def, q):
        # Set a Polyscope camera view which nicely looks at the scene
        
        # Example:
        # (could also be dependent on x)
        # ps.look_at((2., 1., 2.), (0., 0., 0.))
    
        pass
