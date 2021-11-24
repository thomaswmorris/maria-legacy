# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from datetime import datetime
from datetime import timezone
import maria




        
atmosphere_config = {'n_layers'   : 4,         # how many layers to simulate, based on the integrated atmospheric model 
                    'min_depth'   : 50,      # the height of the first layer 
                    'max_depth'   : 3000,      # 
                    'rel_atm_rms' : 1e-1,  
                    'outer_scale' : 500}


pointing_config = {'scan_type' : 'lissajous_box',
                    'duration' : 300,'samp_freq' : 50,
                 'center_azim' : -45, 'center_elev' : 45, 
                     'x_throw' : 5, 'x_period' : 21,
                     'y_throw' : 5, 'y_period' : 29}

pointing_config = {'scan_type' : 'CES',
                    'duration' : 60, 'samp_freq' : 20,
                 'center_azim' : 55, 'center_elev' : 45, 
                    'az_throw' : 15, 'az_speed' : 1.5}

print(dir(maria))

default_model = maria.model()

default_model.sim()


