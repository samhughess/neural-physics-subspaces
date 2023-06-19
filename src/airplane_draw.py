import aerosandbox as asb

# Define airplane parameters for aero calcs
name_airplane = "stoprotor_vtol"
airfoil_airplane = "clarky"
symmetry = True

# # Declare the airplane
# wing_airfoil = asb.Airfoil(airfoil_airplane)
# airplane = asb.Airplane(name = name_airplane,
#                                     xyz_ref = [0, 0, 0], # Cg location
#                                     wings=[
#                                         asb.Wing(
#                                             name = "Right Wing",
#                                             symmetric = False,
#                                             xsecs = [
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.04, 0.05, -0.085],
#                                                     chord = -0.16,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.018, 0.170, -0.085],
#                                                     chord = -0.112,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.0025, 0.3, -0.085],
#                                                     chord = -0.064,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                             ]
#                                         ),
#                                         asb.Wing(
#                                             name = "Left Wing",
#                                             symmetric = False,
#                                             xsecs = [
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.04, -0.05, -0.085],
#                                                     chord = 0.16,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.018, -0.170, -0.085],
#                                                     chord = 0.112,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.0025, -0.3, -0.085],
#                                                     chord = 0.064,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                             ]
#                                         ),
#                                     ],
#                                     fuselages = [
#                                         asb.Fuselage(
#                                             name="Fuselage",
#                                             xsecs = [
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.084, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.025, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.24,
#                                                     shape = 100, 
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.025, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
                                                
                                        
                                                
#                                             ]
#                                         )
#                                     ]
#                                 )

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

airplane.draw()