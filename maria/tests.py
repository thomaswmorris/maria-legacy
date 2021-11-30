# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from datetime import datetime
from datetime import timezone
import maria
        
default_atmosphere_config = {'n_layers'                       : 16,         # how many layers to simulate, based on the integrated atmospheric model 
                                  'min_depth'                 : 50,      # the height of the first layer 
                                  'max_depth'                 : 5000,      # 
                                  'rel_atm_rms'               : 1e-1,  
                                  'turbulence_model'          : 'scale_invariant',
                                  'outer_scale'               : 500}

default_site_config = {'site' : 'ACT',
                       'time' : datetime.now(timezone.utc).timestamp(),
             'weather_gen_method' : 'random',
                     'region' : 'atacama' }


default_array_config = {'shape' : 'hex',
                            'n' : 600,       # maximum number of detectors
                          'fov' : 10,
                    'nom_bands' : 1.5e11,
                  'white_noise' : 0}         # maximum span of array

default_beams_config = {'optical_type' : 'diff_lim',
                          'beam_model' : 'top_hat',
                        'primary_size' : 5.5,
                        'min_beam_res' : .5 }     
     
default_pointing_config = {'scan_type' : 'CES',
                           'duration'  : 10,'samp_freq' : 20,
                         'center_azim' : 0, 'center_elev'  : 90, 
                            'az_throw' : 0, 'az_speed' : 1.5,
                            'el_throw' : 0, 'el_speed' : 1.5}

print(dir(maria))

default_model = maria.model(verbose=True)

default_model.sim()




