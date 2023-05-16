import jax.numpy as jnp
from jax import grad, jit, vmap
#import fixed_point_projection
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as psim

import scipy.constants

import scipy.spatial.transform as transform


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

    # Initialize a variable to calculate the total center of mass
    total_mass = 0

    # Initialize the center of mass coordinates
    center_of_mass = ([0.0, 0.0, 0.0])

    # Iterate over each face of the mesh
    for face in f:
        # Get the vertices of the face
        v0, v1, v2 = v[face]

        # Calculate the area of the face
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Calculate the centroid of the face
        centroid = (v0 + v1 + v2) / 3.0

        # Update the total mass and center of mass
        total_mass += area
        center_of_mass += area * centroid

    # Calculate the final center of mass coordinates
    center_of_mass /= total_mass

    #vol = igl.massmatrix(v,f).data
    #vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    #c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - center_of_mass

    #c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    #c = np.sum(vol[:, None]*v[:vol.shape[0]], axis=0) / np.sum(vol)
    #v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    #mass = np.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [[1, 0, 0],[0, 1, 0],[0, 0, 1], center_of_mass] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': total_mass }
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
        system = Aircraft()

        # Chosen default parameters:
        system.system_name = "Aircraft"
        system.problem_name = str(problem_name)
        
        system_def['cond_params'] = jnp.zeros((0,)) # a length-0 array
        system_def['external_forces'] = {}
        system_def["contact_stiffness"] = 1000000.0
        system.cond_dim = 0
        system.body_ID = None

        bodies = []
        joint_list = []
        numBodiesFixed = 1

        if problem_name == 'generic airplane':
            scale = 1
            
            # Add all the necessary bodies
            bodies.append( make_body( os.path.join(".", "data", "Glider.obj"), 1000, scale))
            
            numBodiesFixed = 1

            # Add all the necessary joints
            #joint_list.append( make_joint(0, -1, bodies, jnp.array([ 0, 0.08 ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ))

            # Define the external forces
            system_def["gravity"] = jnp.array([0.0, 0.0, -9.8])
            system_def['external_forces']['aero_force'] = 0
            system_def['external_forces']['thrust_force'] = 0
            system_def['external_forces']['thrustforce_strength_minmax'] = (0,300)

            system.body_ID = np.array([1])

            #config_dim = 280
            #system_def['dim'] = config_dim
            #system_def['init_pos'] = jnp.zeros(config_dim) # some values
            #system_def['interesting_states'] = jnp.zeros((0,config_dim))
        
        elif problem_name == 'stoprotor':

            # and so on....
            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

        else:
            raise ValueError("could not parse problem name: " + str(problem_name))

        posFixed  = jnp.array( np.array([ body['x0']   for body in bodies[0:numBodiesFixed] ]).flatten() )
        pos  = jnp.array( np.array([ body['x0']   for body in bodies[numBodiesFixed:] ]).flatten() )

        mass = jnp.array( np.array([ body['mass'] for body in bodies[numBodiesFixed:] ]).flatten() )
        
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
        system_def['dim'] = pos.size

        system_def['interesting_states'] = system_def['init_pos'][None,:]

        return system, system_def
  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system

    def potential_energy(self, system_def, q):
        # TODO implement
        qR = q.reshape(-1,4,3)
        massR = system_def['mass'].reshape(-1,4,4)
        gravity = system_def["gravity"]
        c_weighted = massR[:,3,3][:,None]*qR[:,3,:]
        gravity_energy = -jnp.sum(c_weighted * gravity[None,:])
        return gravity_energy
   
    def calculate_thrust_energy(self, thrust_force, distance):
        # Assuming constant thrust force and distance in the direction of the force

        # Calculate work done
        work_done = thrust_force * distance

        # Energy is equal to the work done
        energy = work_done

        return energy
    
    def kinetic_energy(self, system_def, q, q_dot):
        distance = {}
        speed = {}

        qr = q.reshape(-1,4,3)
        #print(qr.shape)
        
        q_dotR = q_dot.reshape(-1,4,3)
        #print(q_dotR.shape)
        massR = system_def['mass'].reshape(-1,4,4)
        
        A = jnp.swapaxes(q_dotR,1,2) @ massR @ q_dotR
        Ke_motion = 0.5*jnp.sum(jnp.trace(A, axis1=1, axis2=2))

        thrust_force =  system_def['external_forces']['thrust_force']
        #distance = distance.append(qr)
        #speed = speed.append(q_dotR)
        #displacement = np.subtract(distance[-1], distance[-2])
        #speed_change = np.subtract(speed[-1], speed[-2])
        Ke_thrust = self.calculate_thrust_energy(thrust_force, qr)
        
        aerodynamic_force = self.compute_aerodynamic_forces(self.bodiesRen, q_dot)
        ke_aero = self.aero_energy(aerodynamic_force, qr, self.mass, q_dotR, q_dotR)
        return np.add(np.add(Ke_motion,Ke_thrust),ke_aero)

    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        # Sample a random, valid setting of the conditional parameters for the system.
        # (If there are no conditional parameters, returning the empty array as below is fine)
        # TODO implement
        return jnp.zeros((0,))

    def aero_energy(self, aerodynamic_force, displacement, mass, initial_velocity, final_velocity):
        # Calculate the work done
        work = np.dot(aerodynamic_force, displacement)

        # Calculate the change in kinetic energy
        delta_kinetic_energy = 0.5 * mass * (final_velocity**2 - initial_velocity**2)

        # Total energy is the sum of work and change in kinetic energy
        energy = work + delta_kinetic_energy

        return energy


    def compute_aerodynamic_forces(self, bodies, q_dot):
        #fluid_density = scipy.constants.density_of_air  # Density of air
        fluid_density = 1.293  # Density of air
        wind_velocity = jnp.array([0.0, 0.0, 0.0])  # Example wind velocity (modify as needed)
        
        q_dotR = q_dot.reshape(-1,4,3)
        air_density = 0.5 * fluid_density * wind_velocity.dot(wind_velocity)

        for body in bodies:
            v = body['v']
            f = body['f']
            W = body['W']
            x0 = body['x0']

            # Compute the relative velocity between the body and the wind
            body_velocity = q_dotR
            relative_velocity_old = body_velocity - wind_velocity
            #relative_velocity = relative_vel.reshape(-1,3)
            relative_velocity = jnp.reshape(relative_velocity_old, (4,3))
            rel_vel = relative_velocity[:,:]
            print(rel_vel.shape)


            # Compute the surface area of each triangle face
            v0 = v[f[:,0]]
            v1 = v[f[:,1]]
            v2 = v[f[:,2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]


            # Calculate the face rotations
            up_vector = np.array([0, 0, 1])
            face_rotations = []
            face_rotation_angles = []
            for normal in face_normals:
                rotation = transform.Rotation.align_vectors(up_vector[np.newaxis, :], normal[np.newaxis, :])[0]
                face_rotations.append(rotation)
                rotation_angle = rotation.as_rotvec()
                face_rotation_angles.append(rotation_angle)
            face_rotations = np.stack(face_rotations)
            face_rotation_angles = np.stack(face_rotation_angles)

            # Calculate the face area
            cross_product = np.cross(v1 - v0, v2 - v0)
            face_area = 0.5 * np.linalg.norm(cross_product, axis=1)
            #face_normals = utils.triangle_normals(v, f)
            #face_areas = utils.triangle_areas(v, f)
            #surface_areas = face_areas.dot(jnp.abs(face_normals))

            # Compute the angle of attack for each face
            #face_rotations = Rotation.align_vectors(jnp.array([0, 0, 1]), cross_product)
            #face_rotations_angles = face_rotations[1].magnitude()

            # Compute the aerodynamic forces for each face
            face_forces = jnp.zeros_like(v)
            for i, face in enumerate(f):
                face_normal = face_normals[i]
                face_areas = face_area[i]
                face_rotation_angle = face_rotation_angles[i]

                # Compute the lift and drag coefficients based on the angle of attack
                #lift_coefficient = self.compute_lift_coefficient(face_rotation_angle)
                #drag_coefficient = self.compute_drag_coefficient(face_rotation_angle)
                lift_coefficient = 1.0
                drag_coefficient = 0.2
                # Compute the lift and drag forces
                lift_force = 0.5 * air_density * face_areas * lift_coefficient * jnp.dot(jnp.dot(jnp.abs(relative_velocity),face_normal),face_normal)
                drag_force = 0.5 * air_density * face_areas * drag_coefficient * jnp.dot(jnp.dot(relative_velocity,face_normal),face_normal)

                # Accumulate the forces for each vertex
                for vertex_index in face:
                    face_forces[vertex_index] += lift_force + drag_force

            # Convert the face forces to vertex forces
            vertex_forces = face_forces / W

            # Apply the aerodynamic forces to the body
            body['external_forces'] += vertex_forces



    # ===========================================
    # === Visualization routines
    # ===========================================

    def build_system_ui(system_def):
        if psim.TreeNode("system UI"):
            psim.TextUnformatted("External forces:")

            if "thrust_force" in system_def["external_forces"]:
                low, high = system_def['external_forces']['thrustforce_strength_minmax']
                _, new_val = psim.SliderFloat("thrust_force", float(system_def['external_forces'][ 'thrust_force']), low, high)
                system_def['external_forces']['thrust_force'] = jnp.array(new_val)

            if "force_strength_y" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_y", float(system_def['external_forces'][ 'force_strength_y']), low, high)
                system_def['external_forces']['force_strength_y'] = jnp.array(new_val)

            if "force_strength_z" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_z", float(system_def['external_forces'][ 'force_strength_z']), low, high)
                system_def['external_forces']['force_strength_z'] = jnp.array(new_val)


            psim.TreePop()

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
