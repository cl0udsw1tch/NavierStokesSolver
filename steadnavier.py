from numbers import Number
import numpy as np
from scipy.sparse import coo_matrix, dia_matrix
from typing import List, Literal, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RectBivariateSpline
import linear_solver
from matplotlib.patches import Rectangle

class NavierStokesSolver:
    
    scheme              : Literal["central_difference", "upwind", "hybrid"]

    L_x                 : float
    L_y                 : float
    n_rows              : int
    n_cols              : int
    phi_B               : float
    gamma               : float
    error               : float
    
    # if u is None
    u_B                 : dict
    v_B                 : dict
    B                   : dict
    
    kinematic_viscosity : float
    u_char              : float
    mass_density        : float
    Re                  : float
    
    alpha_m             : float
    alpha_p             : float
    
    A                   : dict
    b                   : dict
    x                   : dict
    
    _constructor_args_  : dict
    
    A_diag              : dia_matrix
    
    obstacles           : Tuple[Tuple]
    
    def __init__(
        
        self, 
        scheme              : Literal["central_difference", "upwind", "hybrid"] = "central_difference",
        L_x                 : float = 1,
        L_y                 : float = 1,
        n_rows              : int = 0,
        n_cols              : int = 0,
        gamma               : float = 1,
        u                   : Tuple[Number | Callable] | None = None,
        phi_B               : Tuple = (None, None, None, None),
        u_B                 : Tuple = (None, None, None, None),
        v_B                 : Tuple = (None, None, None, None),
        p_B                 : Tuple = (None, None, None, None),
        kinematic_viscosity : float = None,
        Re                  : float = 100,
        u_char              : float = 1,
        mass_density        : float = 1,
        alpha_m             : float = 0.7,
        alpha_p             : float = 0.3,
        obstacle_coords     : List[Tuple[Tuple]] = None
        
        ):
        

        if not n_rows or not n_cols:
            max_cells = 3000
            self.n_rows = int(np.sqrt(max_cells * L_y / L_x))
            self.n_cols = int(np.sqrt(max_cells * L_x / L_y))
        else:
            self.n_rows     = n_rows
            self.n_cols     = n_cols

        self._constructor_args_ = {key: val for key, val in locals().items() if key != 'self'}

        
        self.L_x        = L_x
        self.L_y        = L_y
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.delta_x    = self.L_x / self.n_cols
        self.delta_y    = self.L_y / self.n_rows
        self.scheme     = scheme
        self.alpha_m    = alpha_m
        self.alpha_p    = alpha_p
        self.gamma      = gamma
        
        self.A = { '\phi': None, 'm': None, "p\'": None }
        self.b = { '\phi': None, 'u': None, 'v': None, "p\'": None }
        self.x = { '\phi': None, 'u': None, 'v': None, 'u_f': None, 'v_f': None, 's': None, "p\'": None,"p": None }
        self.A_diag = dia_matrix((self.n_rows * self.n_cols, self.n_rows * self.n_cols))

        self.u_B        = {'e': u_B[0], 'w': u_B[1], 'n': u_B[2], 's': u_B[3]}
        self.v_B        = {'e': v_B[0], 'w': v_B[1], 'n': v_B[2], 's': v_B[3]}
        self.phi_B      = {'e': phi_B[0], 'w': phi_B[1], 'n': phi_B[2], 's': phi_B[3]}
        self.p_B        = {'e': p_B[0], 'w': p_B[1], 'n': p_B[2], 's': p_B[3]}
        self.B          = {'u': self.u_B, 'v': self.v_B, '\phi': self.phi_B, 'p': self.p_B}

                
        if u is None: 
   
            assert(bool(kinematic_viscosity) != bool(Re))
            
            self.L_char                 = L_y
            self.u_char                 = u_char
            self.kinematic_viscosity    = kinematic_viscosity if kinematic_viscosity else self.u_char * self.L_char / Re
            self.mass_density           = mass_density
            self.Re                     = Re if Re else self.u_char * self.L_char / kinematic_viscosity
            self.gamma                  = self.L_char * self.u_char / self.Re

        
        self.x['u_f']   = np.zeros((self.n_rows + 2, self.n_cols + 1))
        self.x['v_f']   = np.zeros((self.n_rows + 1, self.n_cols + 2))        
        self.x['u']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
        self.x['v']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
        self.x['s']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))

        
        self.boundary_mask                  = np.zeros((self.n_rows + 2, self.n_cols + 2)).astype(bool)
        self.boundary_mask[[0, -1], :]      = True
        self.boundary_mask[:, [0, -1]]      = True
        self.interior_mask                  = ~self.boundary_mask
        self.obstacle_mask                  = np.zeros((self.n_rows + 2, self.n_cols + 2)).astype(bool)
        self.obstacles = []

        if obstacle_coords:
            for obs in obstacle_coords:
                (x1, y1), (x2, y2) = self._obstacle_coords_(obs) 
                self.obstacles.append(((x1, y1), (x2, y2)))
                self.obstacle_mask[y1+1:y2+1, x1:x2+1] = True

        self.boundary_mask |= self.obstacle_mask
        self.fluid_mask     = ~self.boundary_mask

        self.e_nbr_mask     = self.fluid_mask & np.roll(self.fluid_mask, 1, 1)
        self.w_nbr_mask     = self.fluid_mask & np.roll(self.fluid_mask, -1, 1)
        self.n_nbr_mask     = self.fluid_mask & np.roll(self.fluid_mask, -1, 0)
        self.s_nbr_mask     = self.fluid_mask & np.roll(self.fluid_mask, 1, 0)

        self.has_e_nbr_mask = self.fluid_mask & np.roll(self.fluid_mask, -1, 1)
        self.has_w_nbr_mask = self.fluid_mask & np.roll(self.fluid_mask, 1, 1)
        self.has_n_nbr_mask = self.fluid_mask & np.roll(self.fluid_mask, 1, 0)
        self.has_s_nbr_mask = self.fluid_mask & np.roll(self.fluid_mask, -1, 0)
        
        self.e_bnd_mask     = self.boundary_mask & np.roll(self.fluid_mask, 1, 1)
        self.w_bnd_mask     = self.boundary_mask & np.roll(self.fluid_mask, -1, 1)
        self.n_bnd_mask     = self.boundary_mask & np.roll(self.fluid_mask, -1, 0)
        self.s_bnd_mask     = self.boundary_mask & np.roll(self.fluid_mask, 1, 0)
        
        self.has_e_bnd_mask = self.fluid_mask & np.roll(self.boundary_mask, -1, 1)
        self.has_w_bnd_mask = self.fluid_mask & np.roll(self.boundary_mask, 1, 1)
        self.has_n_bnd_mask = self.fluid_mask & np.roll(self.boundary_mask, 1, 0)
        self.has_s_bnd_mask = self.fluid_mask & np.roll(self.boundary_mask, -1, 0)

        self.dirichlet_bnd = {
            'm':{}, 'p': {}, '\phi': {}
        }

        phi_dir_e = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        phi_dir_e[:, -1] = False if self.B['\phi']['e'] is None else True
        phi_dir_w = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        phi_dir_w[:, 0] = False if self.B['\phi']['w'] is None else True
        phi_dir_n = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        phi_dir_n[0, :] = False if self.B['\phi']['n'] is None else True
        phi_dir_s = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        phi_dir_s[-1,  :] = False if self.B['\phi']['s'] is None else True
        phi_dir = phi_dir_e | phi_dir_w | phi_dir_n | phi_dir_s

        self.dirichlet_bnd['\phi'] = {
                'all': phi_dir,
                'e': phi_dir_e,
                'w': phi_dir_w,
                'n': phi_dir_n,
                's': phi_dir_s,
            }

        m_dir_e = self.boundary_mask & np.roll(self.fluid_mask, 1, 1)
        m_dir_e[:, -1] = False if self.B['u']['e'] is  None else True
        m_dir_w = self.boundary_mask & np.roll(self.fluid_mask, -1, 1)
        m_dir_w[:, 0] = False if self.B['u']['w'] is  None else True
        m_dir_n = self.boundary_mask & np.roll(self.fluid_mask, -1, 0)
        m_dir_n[0, :] = False if self.B['u']['n'] is  None else True
        m_dir_s = self.boundary_mask & np.roll(self.fluid_mask, 1, 0)
        m_dir_s[-1, :] = False if self.B['u']['s'] is  None else True
        m_dir = m_dir_e | m_dir_w | m_dir_n | m_dir_s

        self.dirichlet_bnd['m'] = {
                'all': m_dir,
                'e': m_dir_e,
                'w': m_dir_w,
                'n': m_dir_n,
                's': m_dir_s
            }
        
        p_dir_e = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        p_dir_e[:, -1] = False if self.B['p']['e'] is None else True
        p_dir_w = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        p_dir_w[:, 0] = False if self.B['p']['w'] is None else True
        p_dir_n = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        p_dir_n[0, :] = False if self.B['p']['n'] is None else True
        p_dir_s = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        p_dir_s[-1,  :] = False if self.B['p']['s'] is None else True
        p_dir = p_dir_e | p_dir_w | p_dir_n | p_dir_s

        self.dirichlet_bnd['p'] = {
                'all': p_dir,
                'e': p_dir_e,
                'w': p_dir_w,
                'n': p_dir_n,
                's': p_dir_s
        }

        self.neumann_bnd = {
            'm':{},  'p': {}, '\phi': {}
        }

        phi_neum_e = self.boundary_mask & np.roll(self.fluid_mask, 1, 1)
        phi_neum_e[:, -1] = False if self.B['\phi']['e'] is not None else True
        phi_neum_w = self.boundary_mask & np.roll(self.fluid_mask, -1, 1)
        phi_neum_w[:, 0] = False if self.B['\phi']['w'] is not None else True
        phi_neum_n = self.boundary_mask & np.roll(self.fluid_mask, -1, 0)
        phi_neum_n[0, :] = False if self.B['\phi']['n'] is not None else True
        phi_neum_s = self.boundary_mask & np.roll(self.fluid_mask, 1, 0)
        phi_neum_s[-1, :] = False if self.B['\phi']['s'] is not None else True
        phi_neum = phi_neum_e | phi_neum_w | phi_neum_n | phi_neum_s

        self.neumann_bnd['\phi'] = {
                'all': phi_neum,
                'e': phi_neum_e,
                'w': phi_neum_w,
                'n': phi_neum_n,
                's': phi_neum_s
            }
        
        m_neum_e = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        m_neum_e[1:-1, -1] = False if self.B['u']['e'] is not None else True
        m_neum_w = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        m_neum_w[1:-1, 0] = False if self.B['u']['w'] is not None else True
        m_neum_n = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        m_neum_n[0, 1:-1] = False if self.B['u']['n'] is not None else True
        m_neum_s = np.zeros((self.n_rows +2, self.n_cols+2)).astype(bool)
        m_neum_s[-1, 1:-1] = False if self.B['u']['s'] is not None else True
        m_neum = m_neum_e | m_neum_w | m_neum_n | m_neum_s


        self.neumann_bnd['m'] = {
            'all': m_neum,
            'e': m_neum_e,
            'w': m_neum_w,
            'n': m_neum_n,
            's': m_neum_s
        }

        p_neum_e = self.boundary_mask & np.roll(self.fluid_mask, 1, 1)
        p_neum_e[:, -1] = False if self.B['p']['e'] is not None else True
        p_neum_w = self.boundary_mask & np.roll(self.fluid_mask, -1, 1)
        p_neum_w[:, 0] = False if self.B['p']['w'] is not None else True
        p_neum_n = self.boundary_mask & np.roll(self.fluid_mask, -1, 0)
        p_neum_n[0, :] = False if self.B['p']['n'] is not None else True
        p_neum_s = self.boundary_mask & np.roll(self.fluid_mask, 1, 0)
        p_neum_s[-1, :] = False if self.B['p']['s'] is not None else True
        p_neum = p_neum_e | p_neum_w | p_neum_n | p_neum_s

        self.neumann_bnd['p'] = {
                'all': p_neum,
                'e': p_neum_e,
                'w': p_neum_w,
                'n': p_neum_n,
                's': p_neum_s
            }
        

        self.e_nbr_face_mask = self.e_nbr_mask[:, 1:]
        self.w_nbr_face_mask = self.w_nbr_mask[:, :-1] # duplicate
        self.n_nbr_face_mask = self.n_nbr_mask[:-1, :]
        self.s_nbr_face_mask = self.s_nbr_mask[1:, :] # duplicate

        self.u_f_fluid_mask = self.e_nbr_face_mask
        self.v_f_fluid_mask = self.n_nbr_face_mask

        self.f_e_mask = (self.has_e_nbr_mask | self.has_e_bnd_mask)[:, :-1]
        self.f_w_mask = (self.has_w_nbr_mask | self.has_w_bnd_mask)[:, 1:]
        self.f_n_mask = (self.has_n_nbr_mask | self.has_n_bnd_mask)[1:, :]
        self.f_s_mask = (self.has_s_nbr_mask | self.has_s_bnd_mask)[:-1, :]
                  
        # diffusive link coefficients
        self.D = {
            'w': self._area_('w') * self.gamma / self.delta_y,
            'e': self._area_('e') * self.gamma / self.delta_y,
            'n': self._area_('n') * self.gamma / self.delta_x,
            's': self._area_('s') * self.gamma / self.delta_x,
        }
        
        self.A['\phi'] = {
                'p': np.zeros((self.n_rows + 2, self.n_cols + 2)),
                'e': np.zeros((self.n_rows + 2, self.n_cols + 2)),
                'w': np.zeros((self.n_rows + 2, self.n_cols + 2)),
                's': np.zeros((self.n_rows + 2, self.n_cols + 2)),
                'n': np.zeros((self.n_rows + 2, self.n_cols + 2)),
            }
        
        # set the solution boundary (Dirichlet)
        
        self.x['\phi']                                       = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
        self.x['\phi'][:, self.n_cols + 1]                   = self.B['\phi']['e'] if self.B['\phi']['e'] is not None else 0
        self.x['\phi'][:, 0]                                 = self.B['\phi']['w'] if self.B['\phi']['w'] is not None else 0
        self.x['\phi'][0, 1: self.n_cols + 1]                = self.B['\phi']['n'] if self.B['\phi']['n'] is not None else 0
        self.x['\phi'][self.n_rows + 1, 1: self.n_cols + 1]  = self.B['\phi']['s'] if self.B['\phi']['s'] is not None else 0
        self.b['\phi']                                       = np.zeros((self.n_rows + 2,  self.n_cols + 2))

        if u is None: # Set the link_coefficient matrices and guess the solutions
            
            # guess the velocity field solution
            self.A['m'] = {
                'p': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'e': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'w': np.ones((self.n_rows + 2, self.n_cols + 2)),
                's': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'n': np.ones((self.n_rows + 2, self.n_cols + 2)),
            }
            self.A['p\''] = {
                'p': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'e': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'w': np.ones((self.n_rows + 2, self.n_cols + 2)),
                's': np.ones((self.n_rows + 2, self.n_cols + 2)),
                'n': np.ones((self.n_rows + 2, self.n_cols + 2)),
            }

            self.x['p']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
            self.x["p\'"]   = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
            
            self.b['u']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
            self.b['v']     = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
            self.b["p\'"]   = np.zeros(((self.n_rows + 2), (self.n_cols + 2)))
            
            # setting the solution boundary
            
            self.x['u'][:, self.n_cols + 1]                         = self.B['u']['e'] if self.B['u']['e'] is not None else 0
            self.x['u'][:, 0]                                       = self.B['u']['w'] if self.B['u']['w'] is not None else 0
            self.x['u'][0, 1: self.n_cols + 1]                      = self.B['u']['n'] if self.B['u']['n'] is not None else 0
            self.x['u'][self.n_rows +1, 1: self.n_cols + 1]         = self.B['u']['s'] if self.B['u']['s'] is not None else 0
            
            self.x['v'][:, self.n_cols + 1]                         = self.B['v']['e'] if self.B['v']['e'] is not None else 0
            self.x['v'][:, 0]                                       = self.B['v']['w'] if self.B['v']['w'] is not None else 0
            self.x['v'][0, 1:self.n_cols + 1]                       = self.B['v']['n'] if self.B['v']['n'] is not None else 0
            self.x['v'][self.n_rows + 1, 1: self.n_cols + 1]        = self.B['v']['s'] if self.B['v']['s'] is not None else 0

            self.x['p'][:, self.n_cols + 1]                         = self.B['p']['e'] if self.B['p']['e'] is not None else 0
            self.x['p'][:, 0]                                       = self.B['p']['w'] if self.B['p']['w'] is not None else 0
            self.x['p'][0, 1:self.n_cols + 1]                       = self.B['p']['n'] if self.B['p']['n'] is not None else 0
            self.x['p'][self.n_rows + 1, 1: self.n_cols + 1]        = self.B['p']['s'] if self.B['p']['s'] is not None else 0

            self.x['u_f'][:, self.n_cols]                           = self.B['u']['e'] if self.B['u']['e'] is not None else 0
            self.x['u_f'][:, 0]                                     = self.B['u']['w'] if self.B['u']['w'] is not None else 0
            self.x['u_f'][0, :]                                     = self.B['u']['n'] if self.B['u']['n'] is not None else 0
            self.x['u_f'][self.n_rows + 1, :]                       = self.B['u']['s'] if self.B['u']['s'] is not None else 0
            
            self.x['v_f'][:, self.n_cols + 1]                       = self.B['v']['e'] if self.B['v']['e'] is not None else 0
            self.x['v_f'][:, 0]                                     = self.B['v']['w'] if self.B['v']['w'] is not None else 0
            self.x['v_f'][0, :]                                     = self.B['v']['n'] if self.B['v']['n'] is not None else 0
            self.x['v_f'][self.n_rows, :]                           = self.B['v']['s'] if self.B['v']['s'] is not None else 0


            # calculate mass influx
            self.influx_map_e = (self.x['u'] < 0) & self.e_bnd_mask
            self.influx_map_w = (self.x['u'] > 0) & self.w_bnd_mask
            self.influx_map_n = (self.x['v'] < 0) & self.n_bnd_mask
            self.influx_map_s = (self.x['v'] > 0) & self.s_bnd_mask
            self.influx_map = self.influx_map_e | self.influx_map_w | self.influx_map_n | self.influx_map_s

            self.influx = (
                (self.x['u'][self.influx_map_e] * self._area_('e')).sum() if self.influx_map_e.any() else 0 +
                (self.x['u'][self.influx_map_w] * self._area_('w')).sum() if self.influx_map_w.any() else 0 +
                (self.x['v'][self.influx_map_n] * self._area_('n')).sum() if self.influx_map_n.any() else 0 +
                (self.x['v'][self.influx_map_s] * self._area_('s')).sum() if self.influx_map_s.any() else 0
            )



            self.outflux_area_e = (self.neumann_bnd['m']['e'] * self._area_('e')).sum()
            self.outflux_area_w = (self.neumann_bnd['m']['w'] * self._area_('w')).sum()
            self.outflux_area_n = (self.neumann_bnd['m']['n'] * self._area_('n')).sum()
            self.outflux_area_s = (self.neumann_bnd['m']['s'] * self._area_('s')).sum()
            self.outflux_area = self.outflux_area_e + self.outflux_area_w + self.outflux_area_n + self.outflux_area_s

        else:

            uxf = lambda i, j : u[0] if isinstance(u[0], int) else u[0](j * self.delta_x, ((self.n_rows + 1) - i - 1)*self.delta_y)   
            uyf = lambda i, j : u[1] if isinstance(u[1], int) else u[1](j * self.delta_x, ((self.n_rows + 1) - i - 1)*self.delta_y) 
              
            for i in range(self.n_rows + 2):
                for j in range(self.n_cols+1):
                    self.x['u_f'][i,j] = uxf(i, j)
            for i in range(self.n_rows + 1):
                for j in range(self.n_cols + 2):
                    self.x['v_f'][i,j] = uyf(i, j)
            
            for i in range(1, self.n_rows+1):
                for j in range(1, self.n_cols + 1):
                    self.x['u'][i,j] = uxf(i+ self.delta_y, j + self.delta_x)
                    self.x['v'][i,j] = uyf(i+ self.delta_y, j + self.delta_x)
                    
            self.x['s'] = np.sqrt(self.x['u']**2 + self.x['v']**2)


    
    def solve(self, objective: Literal['\phi', 'm'] ='\phi', solveSparse=True, tolerance=0.01, max_iter=10_000, threshold=1e-3):
        
        if objective == '\phi':
            self._construct_problem_(objective='\phi')
            self._solve_jacobi_(objective='\phi', tolerance=tolerance, max_iter=max_iter) if not solveSparse else self._solve_sparse_(objective='\phi')
            self._set_homog_neum_phi_()
            self.x['\phi'][self.boundary_mask & ~self.e_bnd_mask & ~self.w_bnd_mask & ~self.n_bnd_mask & ~self.s_bnd_mask] = np.nan
            return self.x['\phi']

        rmse = np.inf
        i = 0
  
        while rmse > threshold:

            u_prev = self.x['u'].copy()
            v_prev = self.x['v'].copy()
            
            self._construct_problem_(objective='m')
            
       
            self._solve_sparse_(objective='u') if solveSparse else self._solve_jacobi_(objective='u', tolerance=tolerance, max_iter=max_iter)
            self._solve_sparse_(objective='v') if solveSparse else self._solve_jacobi_(objective='v', tolerance=tolerance, max_iter=max_iter)

            self._set_homog_neum_uv_()
            #self._set_dirichlet_uv_()
            self._interp_face_v_()
      
            self._construct_problem_(objective="p\'")
            self._solve_sparse_(objective="p\'") if solveSparse else self._solve_jacobi_(objective='p\'', tolerance=tolerance, max_iter=max_iter)
            

            self._correct_p_()
            self._correct_cell_v_()
            #self._correct_face_v_()
            
            self.x['s'] = np.sqrt(self.x['u']**2 + self.x['v']**2)
            
            p_RMSE = np.sqrt(np.mean(np.square(self.x['p\''])))
            u_RMSE = np.sqrt(np.mean(np.square(self.x['u'] - u_prev)))
            v_RMSE = np.sqrt(np.mean(np.square(self.x['v'] - v_prev)))
            
            rmse = u_RMSE + v_RMSE + p_RMSE
            self.error = rmse
            print(f"\rIteration: {i}, u_RMSE: {u_RMSE:.3e}, v_RMSE: {v_RMSE:.3e}, p'_RMSE: {p_RMSE:.3e}", sep='', end='')
            
            
            i+=1
        
        print(f"\nFinished with total error {rmse}")

        for v in [self.x['u'], self.x['v'], self.x['p']]:
            v[self.boundary_mask & ~self.e_bnd_mask & ~self.w_bnd_mask & ~self.n_bnd_mask & ~self.s_bnd_mask] = np.nan

        return self.x['u'], u_RMSE, self.x['v'], v_RMSE, self.x['p'], p_RMSE
            
    def plot(self, 
             objective          : Literal['\phi', 'u', 'v', 'p']=None, 
             levels             : int = 40, 
             name               : str = None, 
             save               : bool = False,
             figsize            : Tuple[int, int] = (7,5),  
             streamlines        : float = 1.5
             ):
        
        core = self.x[objective][1:-1, 1:-1][::-1] if objective else None
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
                    
        x_points = np.arange(0, self.n_cols)
        y_points = np.arange(0, self.n_rows)

        X, Y = np.meshgrid(x_points, y_points)
        
        u = self.x['u'][1:-1, 1:-1][::-1]
        v = self.x['v'][1:-1, 1:-1][::-1]
        
        if objective:
            masked_core = np.ma.masked_where(core == np.nan, core)
            if levels:
                contour = ax.contourf(X, Y, masked_core, levels=levels, corner_mask=False, cmap='jet' if not streamlines else 'bone')
                fig.colorbar(contour, ax=ax).ax.set_title(label=f'${objective}$')
            else:
                sns.heatmap(masked_core, cmap='jet' if not streamlines else 'bone')
                
            if streamlines:
                streamplot = plt.streamplot(X, Y, u, v, density=streamlines, linewidth=1, color=np.sqrt(u**2 + v**2), arrowsize=1, cmap='jet')
                fig.colorbar(streamplot.lines, ax=ax, location = 'left').ax.set_title(label=f'$s$')
                
        else:
            assert(streamlines > 0)
            streamplot = plt.streamplot(X, Y, u, v, density=streamlines, linewidth=1, color=np.sqrt(u**2 + v**2), arrowsize=1, cmap='jet')
            fig.colorbar(streamplot.lines, ax=ax).ax.set_title(label=f'$s$')
                        
        num_ticks = 5
        x_ticks = np.linspace(0, self.n_cols-1, num_ticks)
        x_labels = np.linspace(0, self.L_x, num_ticks)

        y_ticks = np.linspace(0, self.n_rows-1, num_ticks)
        y_labels = np.linspace(0, self.L_y, num_ticks)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, rotation=0)
        ax.margins(x=0,y=0)

        for obstacle in self.obstacles:
            (x1,y1),(x2,y2) = obstacle
            ax.add_patch(Rectangle((x1, self.n_rows - y2), x2-x1-1/2, y2-y1-1/2,  fill='white', hatch='..', zorder=20))
            
        if save:
            plt.savefig(name, dpi=300, bbox_inches="tight")
        
        plt.show()

    # plot vertical speed v along the horizontal centerline
    def plot_vhline(self, vref=None, vref_label="", cols=None):
        
        center_row = (self.n_cols + 2) // 2
        x = np.arange(1, self.n_cols+1) if cols is None else cols
    
        velocity_profile = self.x['v'][center_row, x]

        plt.plot(x, velocity_profile, marker='o', color='b', label="experiment")  
        
        if vref:
            assert(len(vref) == len(x))
            plt.plot(x, vref, marker="o", color='r', label = vref_label)

        plt.title(f"$v$ Profile Along Horizontal Centerline ({self.n_rows}x{self.n_cols} grid, Re={self.Re})")
        plt.xlabel(f"Horizontal Position (Column Number)")
        plt.ylabel(f"$v$")
        plt.grid()
        plt.legend()
        plt.show()
    
    # plot horizontal speed u along the vertical centerline
    def plot_uvline(self, uref=None, uref_label="", rows=None):
                
        center_col = (self.n_rows + 2 ) // 2
        x = np.arange(1, self.n_rows+1)[::-1] if rows is None else rows
    
        velocity_profile = self.x['u'][x, center_col]

        plt.plot(x, velocity_profile, marker='o', color='b', label="experiment")  
        
        if uref:
            assert(len(uref) == len(x))
            plt.plot(x, uref, marker="o", color='r', label = uref_label)

        plt.title(f"$u$ Profile Along Horizontal Centerline ({self.n_rows}x{self.n_cols} grid, Re={self.Re})")
        plt.xlabel(f"Vertical Position (Row Number)")
        plt.ylabel(f"$u$")
        plt.grid()
        plt.legend()
        plt.show()
        
    
    def order_conv(self, 
                   exact_res: float = 320, 
                   coarse_res: float = 80, 
                   fine_res: float = 160,
                   objective: Literal['\phi', 'uvp'] ='\phi') -> float:

        exact_solver = NavierStokesSolver(**{**self._constructor_args_, 'n_cols': exact_res, 'n_rows': exact_res})
        coarse_solver = NavierStokesSolver(**{**self._constructor_args_, 'n_cols': coarse_res, 'n_rows': coarse_res})
        fine_solver = NavierStokesSolver(**{**self._constructor_args_, 'n_cols': fine_res, 'n_rows': fine_res})
        
        if objective == '\phi':
            x_coarse, _ = coarse_solver.solve('\phi')
            x_fine, _ = fine_solver.solve('\phi')
            x_exact, _ = exact_solver.solve('\phi')
            return self._get_order_(x_coarse[1:-1, 1:-1], x_fine[1:-1, 1:-1], x_exact[1:-1, 1:-1], coarse_res, fine_res, exact_res)
        
        elif objective == 'uvp':
            x_coarses = coarse_solver.solve('m')
            x_fines = fine_solver.solve('m')
            x_exacts = exact_solver.solve('m')
            
            o = [None, None, None]
            for i in range(3):
                x_coarse = x_coarses[2*i]
                x_fine = x_fines[2*i]
                x_exact = x_exacts[2*i]
                o[i] = self._get_order_(x_coarse[1:-1, 1:-1], x_fine[1:-1, 1:-1], x_exact[1:-1, 1:-1], coarse_res, fine_res, exact_res)
            return o

    def _get_order_(self, x_coarse, x_fine, x_exact, coarse_res, fine_res, exact_res) -> float:
        
        linsp_coarse = np.linspace(0, 1, coarse_res)
        linsp_fine = np.linspace(0, 1, fine_res)
        linsp_exact = np.linspace(0, 1, exact_res)

        x_coarse_interpolator = RectBivariateSpline(linsp_coarse, linsp_coarse, x_coarse)

        x_coarse_interp = x_coarse_interpolator(linsp_exact, linsp_exact)

        x_fine_interpolator = RectBivariateSpline(linsp_fine, linsp_fine, x_fine)

        x_fine_interp = x_fine_interpolator(linsp_exact, linsp_exact)

        err_coarse = np.linalg.norm(x_exact - x_coarse_interp) / coarse_res
        err_fine = np.linalg.norm(x_exact - x_fine_interp) / fine_res

        O = (np.log(np.abs(err_coarse / err_fine))) / (np.log((1 / coarse_res) / (1 / fine_res)))
        
        return O

    def _construct_problem_(self, objective: Literal["\phi", "m", "p\'"] = "\phi"):

        if objective == 'p\'':
            self._p_corr_coeffs_()
            return
        
        b_e, b_w, b_n, b_s = None, None, None, None
        if self.scheme == "central_difference":
            b_e, b_w, b_n, b_s = self._scalar_coeff_cd_(
                self.A[objective]['p'],
                self.A[objective]['e'],
                self.A[objective]['w'],
                self.A[objective]['n'],
                self.A[objective]['s'],
                objective=objective
                )
        elif self.scheme == 'upwind':
            b_e, b_w, b_n, b_s = self._scalar_coeff_upwind_(
                self.A[objective]['p'],
                self.A[objective]['e'],
                self.A[objective]['w'],
                self.A[objective]['n'],
                self.A[objective]['s'],
                objective=objective
                )
        elif self.scheme == 'hybrid':
            b_e, b_w, b_n, b_s = self._scalar_coeff_hybrid_(
                self.A[objective]['p'],
                self.A[objective]['e'],
                self.A[objective]['w'],
                self.A[objective]['n'],
                self.A[objective]['s'],
                objective=objective
                )
            
        if objective == '\phi':
            self._scalar_src_(b_e, b_w, b_n, b_s, self.b['\phi'], self.x['\phi'])
            self._zero_obstacles_phi_()
        else:
            
            self._scalar_src_(b_e, b_w, b_n, b_s, self.b['u'], self.x['u'], objective='m')
            self._scalar_src_(b_e, b_w, b_n, b_s, self.b['v'], self.x['v'], objective='m')
            
            self.b['u'][self.fluid_mask] -= self._p_src_('u')
            self.b['v'][self.fluid_mask] -= self._p_src_('v')
            
            self.A['m']['p'][self.fluid_mask] /= self.alpha_m
            self.b['u'][self.fluid_mask] += (1 - self.alpha_m) * self.A['m']['p'][self.fluid_mask] * self.x['u'][self.fluid_mask]
            self.b['v'][self.fluid_mask] += (1 - self.alpha_m) * self.A['m']['p'][self.fluid_mask] * self.x['v'][self.fluid_mask]
            
            self._zero_obstacles_m_() if objective == 'm' else self._zero_obstacles_pp_()

    # Correct the pressure field
    def _p_src_(self, objective):

        
        if objective == "u":
            return  (1/2) * (self.x['p'][self.e_nbr_mask | self.e_bnd_mask] - self.x['p'][self.w_nbr_mask | self.w_bnd_mask]) * self._area_('e') 
                    
        elif objective == "v":
            return (1/2) * (self.x['p'][self.n_nbr_mask | self.n_bnd_mask] - self.x['p'][self.s_nbr_mask | self.s_bnd_mask]) * self._area_('n')
        
    
    def _to_sparse_(self, a_E, a_W, a_N, a_S, a_P):
        nrows, ncols = self.n_rows, self.n_cols
        num_cells = nrows * ncols

        a_P_flat = a_P[self.interior_mask] 
        a_E_flat = a_E[self.interior_mask] 
        a_W_flat = a_W[self.interior_mask]  
        a_N_flat = a_N[self.interior_mask] 
        a_S_flat = a_S[self.interior_mask] 

        # Set diagonals directly (matrix flipped along horizontal)
        self.A_diag.setdiag(a_P_flat, k=0)                # Main diagonal
        self.A_diag.setdiag(-a_E_flat[:-1], k=1)          # East diagonal
        self.A_diag.setdiag(-a_W_flat[1:], k=-1)          # West diagonal
        self.A_diag.setdiag(-a_N_flat[ncols:], k=-ncols)  # North diagonal (flipped matrix means the coeff is to the left)
        self.A_diag.setdiag(-a_S_flat[:-ncols], k=ncols)  # South diagonal (flipped matrix means the coeff is to the right)

        return self.A_diag.tocsc()    
            
    def _solve_sparse_(self, objective: Literal["\phi", "m", "p\'"] = "\phi") -> Tuple[np.ndarray, float] :
        
        o = 'm' if objective in ['u', 'v'] else objective
        acsc = self._to_sparse_(
            a_E=self.A[o]['e'],
            a_W=self.A[o]['w'],
            a_N=self.A[o]['n'],
            a_S=self.A[o]['s'],
            a_P=self.A[o]['p']
            )
        bflat = self.b[objective][self.interior_mask]
        x, res = linear_solver.solve(acsc, bflat)
        self.error = res
        self.x[objective][1:-1, 1:-1] = x.reshape((self.n_rows, self.n_cols))
            
        return x, res
    
    def _solve_jacobi_(self, objective: Literal["\phi", "u", "v", "p"] = "\phi", max_iter=10000, tolerance=1e-2) -> Tuple[np.ndarray, float]:
        

        for it in range(max_iter):
            res = 0

            for i in range(1, self.n_rows + 1): 
                for j in range(1, self.n_cols + 1):
 
                    self.x[objective][i, j] = (self.A[objective]['e'][i, j] * self.x[objective][i, j+1] +
                                                self.A[objective]['w'][i, j] * self.x[objective][i, j-1] +
                                                self.A[objective]['n'][i, j] * self.x[objective][i-1, j] +
                                                self.A[objective]['s'][i, j] * self.x[objective][i+1, j] +
                                                self.b[objective][i, j]) / self.A[objective]['p'][i, j]
                    
            for i in range(1, self.n_rows + 1): 
                for j in range(1, self.n_cols + 1):
                    res += np.linalg.norm(self.x[objective][i,j] - (
                        self.A[objective]['e'][i, j] * self.x[objective][i, j+1] + 
                        self.A[objective]['w'][i, j] * self.x[objective][i, j-1] +
                        self.A[objective]['n'][i, j] * self.x[objective][i-1, j] +
                        self.A[objective]['s'][i, j] * self.x[objective][i+1, j] +
                        self.b[objective][i, j]) / self.A[objective]['p'][i, j]
                        )

            print(f"\rIter: {it}, res: {res}", sep='', end='')
 
            if it > 0 and res < tolerance:
                break
                
        self.error = res
        return self.x[objective], res          


    # Central difference scheme
    def _scalar_coeff_cd_(self, A_p, A_e, A_w, A_n, A_s, objective='\phi'):
        
        
        F_x = self.x['u_f'] * self._area_('e')
        F_y = self.x['v_f'] * self._area_('n')

        b_e = np.zeros((self.n_rows+2, self.n_cols+2))
        b_w = np.zeros((self.n_rows+2, self.n_cols+2))
        b_n = np.zeros((self.n_rows+2, self.n_cols+2))
        b_s = np.zeros((self.n_rows+2, self.n_cols+2))
        
        A_e[self.has_e_nbr_mask]                                    = self.D['e']      - (1/2) * F_x[self.u_f_fluid_mask]  
        A_e[self.has_e_bnd_mask]                                    = 0
        b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]     = 2 * self.D['e']  - F_x[self.dirichlet_bnd[objective]['e'][:, 1:]]  
        b_e[np.roll(self.neumann_bnd[objective]['e'], -1, 1)]       = 0
        
        A_w[self.has_w_nbr_mask]                                    =  self.D['w']      + (1/2) * F_x[self.u_f_fluid_mask]  
        A_w[self.has_w_bnd_mask]                                    =  0
        b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]      =  2 * self.D['w']  + F_x[self.dirichlet_bnd[objective]['w'][:, :-1]]
        b_w[np.roll(self.neumann_bnd[objective]['w'], 1, 1)]        = 0
        
        A_n[self.has_n_nbr_mask]                                    =  self.D['n']      - (1/2) * F_y[self.v_f_fluid_mask]  
        A_n[self.has_n_bnd_mask]                                    =  0
        b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]      =  2 * self.D['n']  - F_y[self.dirichlet_bnd[objective]['n'][:-1, :]] 
        b_n[np.roll(self.neumann_bnd[objective]['n'], 1, 0)]        = 0
        
        A_s[self.has_s_nbr_mask]                                    =  self.D['s']      + (1/2) * F_y[self.v_f_fluid_mask] 
        A_s[self.has_s_bnd_mask]                                    =  0
        b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]     =  2 * self.D['s']  + F_y[self.dirichlet_bnd[objective]['s'][1:, :]]
        b_s[np.roll(self.neumann_bnd[objective]['s'], -1, 0)]       = 0
        
        A_p[self.fluid_mask] = A_e[self.fluid_mask] + A_w[self.fluid_mask] + A_n[self.fluid_mask] + A_s[self.fluid_mask]  + F_x[self.f_e_mask] - F_x[self.f_w_mask] + F_y[self.f_n_mask] - F_y[self.f_s_mask]


        A_p[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]    += b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]    += b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]    += b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]
        A_p[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]    += b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]
        
        A_p[self.boundary_mask] = 1
        A_w[self.boundary_mask] = 0
        A_e[self.boundary_mask] = 0
        A_n[self.boundary_mask] = 0
        A_s[self.boundary_mask] = 0
        
        return b_e, b_w, b_n, b_s
    
    # Upwind scheme
    def _scalar_coeff_upwind_(self, A_p, A_e, A_w, A_n, A_s, objective='\phi'):
        F_x = self.x['u_f'] * self._area_('e')
        F_y = self.x['v_f'] * self._area_('n')

        b_e = np.zeros((self.n_rows+2, self.n_cols+2))
        b_w = np.zeros((self.n_rows+2, self.n_cols+2))
        b_n = np.zeros((self.n_rows+2, self.n_cols+2))
        b_s = np.zeros((self.n_rows+2, self.n_cols+2))
        
        A_e[self.has_e_nbr_mask]                                    = self.D['e']      + np.maximum(0, -F_x[self.u_f_fluid_mask])
        A_e[self.has_e_bnd_mask]                                    = 0
        b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]     = 2 * self.D['e']  + np.maximum(0, -F_x[self.dirichlet_bnd[objective]['e'][:, 1:]])
        b_e[np.roll(self.neumann_bnd[objective]['e'], -1, 1)]       = 0
        
        A_w[self.has_w_nbr_mask]                                    = self.D['w']      + np.maximum(0, F_x[self.u_f_fluid_mask]) 
        A_w[self.has_w_bnd_mask]                                    = 0
        b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]      = 2 * self.D['w']  + np.maximum(0, F_x[self.dirichlet_bnd[objective]['w'][:, :-1]])
        b_w[np.roll(self.neumann_bnd[objective]['w'], 1, 1)]        = 0
        
        A_n[self.has_n_nbr_mask]                                    = self.D['n']      + np.maximum(0, -F_y[self.v_f_fluid_mask]  )
        A_n[self.has_n_bnd_mask]                                    = 0
        b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]      = 2 * self.D['n']  + np.maximum(0, -F_y[self.dirichlet_bnd[objective]['n'][:-1, :]])
        b_n[np.roll(self.neumann_bnd[objective]['n'], 1, 0)]        = 0
        
        A_s[self.has_s_nbr_mask]                                    = self.D['s']      + np.maximum(0, F_y[self.v_f_fluid_mask])
        A_s[self.has_s_bnd_mask]                                    = 0
        b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]     = 2 * self.D['s']  + np.maximum(0, F_y[self.dirichlet_bnd[objective]['s'][1:, :]])
        b_s[np.roll(self.neumann_bnd[objective]['s'], -1, 0)]       = 0
        
        A_p[self.fluid_mask] = A_e[self.fluid_mask] + A_w[self.fluid_mask] + A_n[self.fluid_mask] + A_s[self.fluid_mask]  + F_x[self.f_e_mask] - F_x[self.f_w_mask] + F_y[self.f_n_mask] - F_y[self.f_s_mask]


        A_p[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]    += b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]    += b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]    += b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]
        A_p[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]    += b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]
        
        A_p[self.boundary_mask] = 1
        A_w[self.boundary_mask] = 0
        A_e[self.boundary_mask] = 0
        A_n[self.boundary_mask] = 0
        A_s[self.boundary_mask] = 0
        
        return b_e, b_w, b_n, b_s
        
    # Hybrid scheme
    def _scalar_coeff_hybrid_(self, A_p, A_e, A_w, A_n, A_s, objective='\phi'):
        F_x = self.x['u_f'] * self._area_('e')
        F_y = self.x['v_f'] * self._area_('n')

        b_e = np.zeros((self.n_rows+2, self.n_cols+2))
        b_w = np.zeros((self.n_rows+2, self.n_cols+2))
        b_n = np.zeros((self.n_rows+2, self.n_cols+2))
        b_s = np.zeros((self.n_rows+2, self.n_cols+2))
        
        A_e[self.has_e_nbr_mask]                                    = np.maximum(self.D['e'] - (1/2) * F_x[self.u_f_fluid_mask], np.maximum(0, -F_x[self.u_f_fluid_mask]))
        A_e[self.has_e_bnd_mask]                                    = 0
        b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]     = np.maximum(self.D['e'] - (1/2) * F_x[self.dirichlet_bnd[objective]['e'][:, 1:]], np.maximum(0, -F_x[self.dirichlet_bnd[objective]['e'][:, 1:]]))
        b_e[np.roll(self.neumann_bnd[objective]['e'], -1, 1)]       = 0
        
        A_w[self.has_w_nbr_mask]                                    = np.maximum(self.D['w'] + (1/2) * F_x[self.u_f_fluid_mask], np.maximum(0, F_x[self.u_f_fluid_mask]))
        A_w[self.has_w_bnd_mask]                                    = 0
        b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]      = np.maximum(2 * self.D['w'] + (1/2) * F_x[self.dirichlet_bnd[objective]['w'][:, :-1]], np.maximum(0, F_x[self.dirichlet_bnd[objective]['w'][:, :-1]]))
        b_w[np.roll(self.neumann_bnd[objective]['w'], 1, 1)]        = 0
        
        A_n[self.has_n_nbr_mask]                                    = np.maximum(self.D['n'] - (1/2) * F_y[self.v_f_fluid_mask], np.maximum(0, -F_y[self.v_f_fluid_mask]))
        A_n[self.has_n_bnd_mask]                                    = 0
        b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]      = np.maximum(2 * self.D['n'] - (1/2) * F_y[self.dirichlet_bnd[objective]['n'][:-1, :]], np.maximum(0, -F_y[self.dirichlet_bnd[objective]['n'][:-1, :]]))
        b_n[np.roll(self.neumann_bnd[objective]['n'], 1, 0)]        = 0
        
        A_s[self.has_s_nbr_mask]                                    = np.maximum(self.D['s'] + (1/2) * F_y[self.v_f_fluid_mask], np.maximum(0, F_y[self.v_f_fluid_mask]))
        A_s[self.has_s_bnd_mask]                                    = 0
        b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]     = np.maximum(2 * self.D['s'] + (1/2) * F_y[self.dirichlet_bnd[objective]['s'][1:, :]], np.maximum(0, F_y[self.dirichlet_bnd[objective]['s'][1:, :]]))
        b_s[np.roll(self.neumann_bnd[objective]['s'], -1, 0)]       = 0
        
        A_p[self.fluid_mask] = A_e[self.fluid_mask] + A_w[self.fluid_mask] + A_n[self.fluid_mask] + A_s[self.fluid_mask]  + F_x[self.f_e_mask] - F_x[self.f_w_mask] + F_y[self.f_n_mask] - F_y[self.f_s_mask]


        A_p[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]    += b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]    += b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)]
        A_p[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]    += b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)]
        A_p[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]    += b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)]
        
        A_p[self.boundary_mask] = 1
        A_w[self.boundary_mask] = 0
        A_e[self.boundary_mask] = 0
        A_n[self.boundary_mask] = 0
        A_s[self.boundary_mask] = 0
        
        return b_e, b_w, b_n, b_s

    # Source term
    def _scalar_src_(self, b_e, b_w, b_n, b_s, b, x, objective='\phi'):
        
        b[:, :]         = 0
        
        b[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)] += b_e[np.roll(self.dirichlet_bnd[objective]['e'], -1, 1)] * x[self.dirichlet_bnd[objective]['e']]
        b[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)] += b_w[np.roll(self.dirichlet_bnd[objective]['w'], 1, 1)] * x[self.dirichlet_bnd[objective]['w']]
        b[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)] += b_n[np.roll(self.dirichlet_bnd[objective]['n'], 1, 0)] * x[self.dirichlet_bnd[objective]['n']]
        b[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)] += b_s[np.roll(self.dirichlet_bnd[objective]['s'], -1, 0)] * x[self.dirichlet_bnd[objective]['s']]
         
    # Rhie-Chow interpolation of face velocities from cell-centered velocities
    def _interp_face_v_(self):
        
        
        a_e = self.A['m']['p'][self.e_nbr_mask] 
        a_p = self.A['m']['p'][self.has_e_nbr_mask] 

        u_e = self.x['u'][self.e_nbr_mask]  
        u_p = self.x['u'][self.has_e_nbr_mask] 

        p_ee = self.x['p'][np.roll(self.e_nbr_mask, 1, axis=1)]       
        p_e = self.x['p'][self.e_nbr_mask] 
        p_p = self.x['p'][self.has_e_nbr_mask]
        p_w = self.x['p'][np.roll(self.has_e_nbr_mask, -1, 1)]


        d_e = (self._area_('e') / a_e) * self.alpha_m
        d_p = (self._area_('e') / a_p) * self.alpha_m


        self.x['u_f'][self.u_f_fluid_mask] = (
            (1 / 2) * (u_e + u_p)
            - (1 / 2) * (d_e + d_p) * (p_e - p_p)
            + (1 / 2) * d_p * ((1 / 2) * (p_e - p_w))
            + (1 / 2) * d_e * ((1 / 2) * (p_ee - p_p))
        )


        a_s = self.A['m']['p'][self.s_nbr_mask]
        a_p = self.A['m']['p'][self.has_s_nbr_mask]

        v_s = self.x['v'][self.s_nbr_mask]
        v_p = self.x['v'][self.has_s_nbr_mask]

        p_n = self.x['p'][np.roll(self.has_s_nbr_mask, -1, 0)] 
        p_p = self.x['p'][self.has_s_nbr_mask]
        p_s = self.x['p'][self.s_nbr_mask] 
        p_ss = self.x['p'][np.roll(self.s_nbr_mask, 1, axis=0)]


        d_s = (self._area_('s') / a_s) * self.alpha_m
        d_p = (self._area_('s') / a_p) * self.alpha_m

   
        self.x['v_f'][self.v_f_fluid_mask] = (
            (1 / 2) * (v_s + v_p)
            - (1 / 2) * (d_s + d_p) * (p_p - p_s)
            + (1 / 2) * d_p * ((1 / 2) * (p_n - p_s))
            + (1 / 2) * d_s * ((1 / 2) * (p_p - p_ss))
        )

    # Pressure correction coefficients
    def _p_corr_coeffs_(self):

        a_p = self.A['m']['p'][self.fluid_mask]
        self.A['m']['p'][self.e_bnd_mask | self.w_bnd_mask | self.n_bnd_mask | self.s_bnd_mask] = 0
        a_e = self.A['m']['p'][self.e_nbr_mask | self.e_bnd_mask]
        a_w = self.A['m']['p'][self.w_nbr_mask | self.w_bnd_mask]
        a_n = self.A['m']['p'][self.n_nbr_mask | self.n_bnd_mask]
        a_s = self.A['m']['p'][self.s_nbr_mask | self.s_bnd_mask]

        d_px = self._area_('e') / a_p
        d_py = self._area_('n') / a_p

        d_e = np.where(a_e != 0, (1/2) * (self._area_('e') / a_e + d_px), 0)
        d_w = np.where(a_w != 0, (1/2) * (self._area_('w') / a_w + d_px), 0)
        d_n = np.where(a_n != 0, (1/2) * (self._area_('n') / a_n + d_py), 0)
        d_s = np.where(a_s != 0, (1/2) * (self._area_('s') / a_s + d_py), 0)

        # Update A['p\''] matrices for 'e', 'w', 'n', 's'
        self.A['p\'']['e'][self.fluid_mask] = self.alpha_m * d_e * self._area_('e')
        self.A['p\'']['w'][self.fluid_mask] = self.alpha_m * d_w * self._area_('w')
        self.A['p\'']['n'][self.fluid_mask] = self.alpha_m * d_n * self._area_('n')
        self.A['p\'']['s'][self.fluid_mask] = self.alpha_m * d_s * self._area_('s')
        
        # Compute central coefficient (sum of neighbor coefficients)
        self.A['p\'']['p'][self.fluid_mask] = (
            self.A['p\'']['e'][self.fluid_mask]
            + self.A['p\'']['w'][self.fluid_mask]
            + self.A['p\'']['n'][self.fluid_mask]
            + self.A['p\'']['s'][self.fluid_mask]
        )
        self.A['m']['p'][self.e_bnd_mask | self.w_bnd_mask | self.n_bnd_mask | self.s_bnd_mask] = 1

        # Compute b['p\''], dependent on face velocities
        uf_e = self.x['u_f'][self.f_e_mask]
        uf_w = self.x['u_f'][self.f_w_mask]
        uf_n = self.x['v_f'][self.f_n_mask]
        uf_s = self.x['v_f'][self.f_s_mask]

        # =============================RHIE_CHOW_DEPENDENT================================ #
        
        self.b['p\''][self.fluid_mask] =- uf_e * self._area_('e')
        self.b['p\''][self.fluid_mask] += uf_w * self._area_('w') 
        self.b['p\''][self.fluid_mask] -= uf_n * self._area_('n')
        self.b['p\''][self.fluid_mask] += uf_s * self._area_('s')

        # =============================RHIE_CHOW_DEPENDENT================================ #


        self.A['p\'']['p'][np.roll(self.dirichlet_bnd['p']['e'], -1, 1)] += 1e30
        self.b['p\''][np.roll(self.dirichlet_bnd['p']['e'], -1, 1)] = 0

        self.A['p\'']['p'][np.roll(self.dirichlet_bnd['p']['w'], 1, 1)] += 1e30
        self.b['p\''][np.roll(self.dirichlet_bnd['p']['w'], 1, 1)] = 0

        self.A['p\'']['p'][np.roll(self.dirichlet_bnd['p']['n'], -1, 0)] += 1e30
        self.b['p\''][np.roll(self.dirichlet_bnd['p']['n'], -1, 0)] = 0

        self.A['p\'']['p'][np.roll(self.dirichlet_bnd['p']['s'], 1, 0)] += 1e30
        self.b['p\''][np.roll(self.dirichlet_bnd['p']['s'], 1, 0)] = 0



        self.A['p\'']['p'][self.boundary_mask] = 1
        self.A['p\'']['e'][self.boundary_mask] = 0
        self.A['p\'']['w'][self.boundary_mask] = 0
        self.A['p\'']['n'][self.boundary_mask] = 0
        self.A['p\'']['s'][self.boundary_mask] = 0
        self.b['p\''][self.boundary_mask]      = 0
     
    # Correct the pressure field
    def _correct_p_(self):

        self.x['p'] += self.alpha_p * self.x["p\'"]
        self._set_homog_dirichlet_p_()
        self._set_homog_neum_p_()

    # Correct cell-centered velocities
    def _correct_cell_v_(self):
        
        a_p = self.A['m']['p'][self.fluid_mask]
        p_e = self.x['p\''][self.e_nbr_mask | self.e_bnd_mask]
        p_w = self.x['p\''][self.w_nbr_mask | self.w_bnd_mask]
        p_n = self.x['p\''][self.n_nbr_mask | self.n_bnd_mask]
        p_s = self.x['p\''][self.s_nbr_mask | self.s_bnd_mask]
        
        d_p = self.alpha_m / a_p
        
        self.x['u'][self.fluid_mask] -= self._area_('w') * d_p * (1/2) * (p_e - p_w)
        self.x['v'][self.fluid_mask] -= self._area_('n') * d_p * (1/2) * (p_n - p_s)
        
    # Correct face velocities (not used in this simulation)
    def _correct_face_v_(self):
        
        a_e = self.A['m']['p'][self.e_nbr_mask]  
        a_p = self.A['m']['p'][self.has_e_nbr_mask]

        p_e = self.x['p\''][self.e_nbr_mask]     
        p_p = self.x['p\''][self.has_e_nbr_mask] 
        
        d_p = (self._area_('e') * self.alpha_m / a_p) 
        d_e = (self._area_('e') * self.alpha_m / a_e) 

        self.x['u_f'][self.u_f_fluid_mask] -= (1/2) * (d_e + d_p) * (p_e - p_p)
            
  
        a_s = self.A['m']['p'][self.s_nbr_mask]
        a_p = self.A['m']['p'][self.has_s_nbr_mask]
        
        p_s = self.x['p\''][self.s_nbr_mask]    
        p_p = self.x['p\''][self.has_s_nbr_mask] 
        
        d_p = (self._area_('s') * self.alpha_m / a_p) 
        d_s = (self._area_('s') * self.alpha_m / a_s) 

        self.x['v_f'][self.v_f_fluid_mask] -= (1/2) * (d_s + d_p) * (p_p - p_s)

    # =================== Boundary Conditions =================== #
    def _set_homog_dirichlet_p_(self):
        self.x['p'][np.roll(self.dirichlet_bnd['p']['e'], -1, 1)] = self.B['p']['e']  if self.B['p']['e'] is not None else 0
        self.x['p'][np.roll(self.dirichlet_bnd['p']['w'],1, 1)] = self.B['p']['w']   if self.B['p']['w'] is not None else 0  
        self.x['p'][np.roll(self.dirichlet_bnd['p']['n'],1, 0)] = self.B['p']['n']  if self.B['p']['n'] is not None else 0 
        self.x['p'][np.roll(self.dirichlet_bnd['p']['s'], -1, 0)] = self.B['p']['s']  if self.B['p']['s'] is not None else 0


    def _set_homog_dirichlet_p_corr_(self):
        self.x['p\''][np.roll(self.dirichlet_bnd['p']['e'], -1, 1)] = 0   
        self.x['p\''][np.roll(self.dirichlet_bnd['p']['w'],1, 1)] = 0    
        self.x['p\''][np.roll(self.dirichlet_bnd['p']['n'],1, 0)] = 0   
        self.x['p\''][np.roll(self.dirichlet_bnd['p']['s'], -1, 0)] = 0
        
    def _set_homog_neum_p_(self):

        self.x['p'][self.neumann_bnd['p']['e']] = self.x['p'][np.roll(self.neumann_bnd['p']['e'], -1, 1)]
        self.x['p'][self.neumann_bnd['p']['w']] = self.x['p'][np.roll(self.neumann_bnd['p']['w'], 1, 1)]
        self.x['p'][self.neumann_bnd['p']['n']] = self.x['p'][np.roll(self.neumann_bnd['p']['n'], 1, 0)]
        self.x['p'][self.neumann_bnd['p']['s']] = self.x['p'][np.roll(self.neumann_bnd['p']['s'], -1, 0)]


    def _set_homog_neum_p_corr_(self):
        self.x['p\''][self.neumann_bnd['p']['e']] = self.x['p\''][np.roll(self.neumann_bnd['p']['e'], -1, 1)]    
        self.x['p\''][self.neumann_bnd['p']['w']] = self.x['p\''][np.roll(self.neumann_bnd['p']['w'], 1, 1)]    
        self.x['p\''][self.neumann_bnd['p']['n']] = self.x['p\''][np.roll(self.neumann_bnd['p']['n'], 1, 0)]    
        self.x['p\''][self.neumann_bnd['p']['s']] = self.x['p\''][np.roll(self.neumann_bnd['p']['s'], -1, 0)]
        
    def _set_homog_neum_phi_(self):
        self.x['\phi'][self.neumann_bnd['\phi']['e']] = self.x['\phi'][np.roll(self.neumann_bnd['\phi']['e'], -1, 1)]
        self.x['\phi'][self.neumann_bnd['\phi']['w']] = self.x['\phi'][np.roll(self.neumann_bnd['\phi']['w'], 1, 1)]
        self.x['\phi'][self.neumann_bnd['\phi']['n']] = self.x['\phi'][np.roll(self.neumann_bnd['\phi']['n'], 1, 0)]
        self.x['\phi'][self.neumann_bnd['\phi']['s']] = self.x['\phi'][np.roll(self.neumann_bnd['\phi']['s'], -1, 0)]

    def _set_homog_neum_uv_(self):

        outflux_e = (self.x['u'][np.roll(self.neumann_bnd['m']['e'], -1, 1)] ).sum() * self._area_('e')
        fix_e = (self.influx / outflux_e) * (self.outflux_area_e / self.outflux_area) if np.any(self.neumann_bnd['m']['e']) else 0
        outflux_w = (self.x['u'][np.roll(self.neumann_bnd['m']['w'], 1, 1)] * self._area_('w')).sum()
        fix_w = (self.influx / outflux_w ) * (self.outflux_area_w / self.outflux_area) if np.any(self.neumann_bnd['m']['w']) else 0
        outflux_n = (self.x['v'][np.roll(self.neumann_bnd['m']['n'], 1, 0)] * self._area_('n')).sum()
        fix_n = (self.influx / outflux_n) * (self.outflux_area_n / self.outflux_area) if np.any(self.neumann_bnd['m']['n']) else 0
        outflux_s = (self.x['v'][np.roll(self.neumann_bnd['m']['s'], -1, 0)] * self._area_('s')).sum()
        fix_s = (self.influx / outflux_s) * (self.outflux_area_s / self.outflux_area) if np.any(self.neumann_bnd['m']['s']) else 0

        self.x['u'][self.neumann_bnd['m']['e']] = self.x['u'][np.roll(self.neumann_bnd['m']['e'], -1, 1)] * fix_e
        self.x['v'][self.neumann_bnd['m']['e']] = self.x['v'][np.roll(self.neumann_bnd['m']['e'], -1, 1)]
        self.x['u_f'][self.neumann_bnd['m']['e'][:, 1:]] = self.x['u'][self.neumann_bnd['m']['e']]
        
        
        self.x['u'][self.neumann_bnd['m']['w']] = self.x['u'][np.roll(self.neumann_bnd['m']['w'], 1, 1)] * fix_w
        self.x['v'][self.neumann_bnd['m']['w']] = self.x['v'][np.roll(self.neumann_bnd['m']['w'], 1, 1)]
        self.x['u_f'][self.neumann_bnd['m']['w'][:, :-1]] = self.x['u'][self.neumann_bnd['m']['w']]
        
        self.x['u'][self.neumann_bnd['m']['n']] = self.x['u'][np.roll(self.neumann_bnd['m']['n'], 1, 0)]
        self.x['v'][self.neumann_bnd['m']['n']] = self.x['v'][np.roll(self.neumann_bnd['m']['n'], 1, 0)] * fix_n
        self.x['v_f'][self.neumann_bnd['m']['n'][:-1, :]] = self.x['v'][self.neumann_bnd['m']['n']]
        
        self.x['u'][self.neumann_bnd['m']['s']] = self.x['u'][np.roll(self.neumann_bnd['m']['s'], -1, 0)]
        self.x['v'][self.neumann_bnd['m']['s']] = self.x['v'][np.roll(self.neumann_bnd['m']['s'], -1, 0)] * fix_s
        self.x['v_f'][self.neumann_bnd['m']['s'][1:, :]] = self.x['v'][self.neumann_bnd['m']['s']]

    def _set_dirichlet_uv_(self):
        self.x['u'][np.roll(self.dirichlet_bnd['m']['e'], -1, 1)] = self.x['u'][self.dirichlet_bnd['m']['e']]
        self.x['u'][np.roll(self.dirichlet_bnd['m']['w'], 1, 1)] = self.x['u'][self.dirichlet_bnd['m']['w']]
        self.x['u'][np.roll(self.dirichlet_bnd['m']['n'], 1, 0)] = self.x['u'][self.dirichlet_bnd['m']['n']]
        self.x['u'][np.roll(self.dirichlet_bnd['m']['s'], -1, 0)] = self.x['u'][self.dirichlet_bnd['m']['s']]
        self.x['v'][np.roll(self.dirichlet_bnd['m']['e'], -1, 1)] = self.x['v'][self.dirichlet_bnd['m']['e']]
        self.x['v'][np.roll(self.dirichlet_bnd['m']['w'], 1, 1)] = self.x['v'][self.dirichlet_bnd['m']['w']]
        self.x['v'][np.roll(self.dirichlet_bnd['m']['n'], 1, 0)] = self.x['v'][self.dirichlet_bnd['m']['n']]
        self.x['v'][np.roll(self.dirichlet_bnd['m']['s'], -1, 0)] = self.x['v'][self.dirichlet_bnd['m']['s']]

    # =================== Boundary Conditions =================== #

    # =================== Obstacle Handling =================== #

    def _zero_obstacles_m_(self):
            self.A['m']['p'][self.boundary_mask] = 1
            self.A['m']['e'][self.boundary_mask] = 0
            self.A['m']['w'][self.boundary_mask] = 0
            self.A['m']['n'][self.boundary_mask] = 0
            self.A['m']['s'][self.boundary_mask] = 0
            self.b['u'][self.boundary_mask] = 0
            self.b['v'][self.boundary_mask] = 0
            
    def _zero_obstacles_pp_(self):
            self.A['m']['p'][self.boundary_mask] = 1
            self.A['m']['e'][self.boundary_mask] = 0
            self.A['m']['w'][self.boundary_mask] = 0
            self.A['m']['n'][self.boundary_mask] = 0
            self.A['m']['s'][self.boundary_mask] = 0
            self.b['p\''][self.boundary_mask] = 0
            
    def _zero_obstacles_phi_(self):
            self.A['\phi']['p'][self.boundary_mask] = 1
            self.A['\phi']['e'][self.boundary_mask] = 0
            self.A['\phi']['w'][self.boundary_mask] = 0
            self.A['\phi']['n'][self.boundary_mask] = 0
            self.A['\phi']['s'][self.boundary_mask] = 0
            self.b['\phi'][self.boundary_mask] = 0

    # =================== Obstacle Handling =================== #
        
    def _area_(self, face):
        if face == 'e' or face == 'w':
            return self.delta_y
        return self.delta_x