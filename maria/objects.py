
import ephem
import scipy as sp
import numpy as np
import pandas as pd
import pickle
    
from scipy import signal,stats
from datetime import datetime
from datetime import timezone
from . import tools
#import tools

from importlib import resources

import weathergen


#print('local objects')

# an array object describes the array of detectors. all of the arguments are dimensionful arrays of the same length 



default_atmosphere_config = {'n_layers'                       : 16,         # how many layers to simulate, based on the integrated atmospheric model 
                                  'min_depth'                 : 100,      # the height of the first layer 
                                  'max_depth'                 : 5000,      # 
                                  'rel_atm_rms'               : 1e-1,  
                                  'turbulence_model'          : 'scale_invariant',
                                  'outer_scale'               : 500}

default_site_config = {'site' : 'ACT',
                       'time' : datetime.now(timezone.utc).timestamp(),
             'weather_gen_method' : 'random',
                     'region' : 'atacama' }


default_array_config = {'shape' : 'hex',
                            'n' : 271,       # maximum number of detectors
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


def validate_config(args,needed_args,name=''):
    
    for arg in needed_args:
        
        assert arg in args, f'Error: {name} config missing argument: {arg}'
    

class atmosphere():
    
    def __init__(self, config=None):
        
        if config==None:
            print('No atm config specified, using 16 geometrically-spaced layers.')
            self.config = default_atmosphere_config.copy()
        else:
            self.config = config.copy()
        
        use_auto_depths   = np.all(np.isin(['min_depth','max_depth','n_layers'],list(self.config)))
        use_manual_depths = np.all(np.isin(['depths'],list(self.config)))
        
        if use_manual_depths: 
            
            if isinstance(self.depths, np.ndarray):
                self.config['min_depth'] = self.depths.min()
                self.config['max_depth'] = self.depths.max()
                self.config['n_layers']  = len(self.depths)
            else:
                raise Exception('\'depths\' parameter must be a numpy array.') 
                
        if not (use_auto_depths or use_manual_depths):
            for arg in ['min_depth','max_depth','n_layers']:
                self.config[arg] = default_atmosphere_config[arg]
            use_auto_depths = True
        if use_auto_depths:  
            #self.depths = np.linspace(self.config['min_depth'], self.config['max_depth'], self.config['n_layers'])
            self.depths = np.geomspace(self.config['min_depth'], self.config['max_depth'], self.config['n_layers'])
            self.thicks = np.gradient(self.depths)
            
            #raise Exception('Could not build atmospheric layers. Please specify the \'min_depth\', \'max_depth\', and \'n_layers\' parameters, or else enter an array of heights.')
        
        necessary_args = ['turbulence_model','outer_scale','rel_atm_rms']
        for arg in necessary_args:
            if not arg in list(self.config):
                self.config[arg] = default_atmosphere_config[arg]
                
        if self.config['turbulence_model'] == 'scale_invariant':
            
            self.matern = lambda r,r0,nu : 2**(1-nu)/sp.special.gamma(nu)*sp.special.kv(nu,r/r0+1e-10)*(r/r0+1e-10)**nu
            
        with resources.path('maria','am_dict.npy') as handle:
            self.spectra_dict = np.load(handle,allow_pickle=True)[()]

            
class site():
    
    def __init__(self, config=None):
        
        if config==None:
            print('No site config specified, defaulting to Cerro Toco.')
            self.config = default_site_config.copy()
        else:
            self.config = config.copy()
        
        if 'site' in list(self.config):
            
            with resources.path("maria", "site_info.csv") as f:
                self.site_df = pd.read_csv(f, index_col=0)
    
            site_list = '\n\nsite' + 5*' ' + 'region' + 7*' ' + 'weather' +  3*' ' + 'longitude' + 3*' ' + 'latitude' + 2*' ' + 'height'
            site_list += '\n' + (len(site_list)-2)*'#'
            
            for sitename in list(self.site_df.index):
                name,sup,loc,lon,lat,hgt = [self.site_df.loc[sitename,key] for key in ['longname','supported','region','longitude','latitude','altitude']]
                lon_name = f'{np.round(np.abs(lon),3):>8}°' + ['W','E'][int(lon>0)]
                lat_name = f'{np.round(np.abs(lat),3):>8}°' + ['S','N'][int(lat>0)]
                site_list += f'\n{sitename:<8} {loc:<12} {sup:<8} {lon_name} {lat_name} {hgt:>6.0f}m'
                
            if not self.config['site'] in self.site_df.index:
                raise Exception('\'' + self.config['site'] + '\' is not a supported site! Supported sites are:' + site_list)
                
            site_info = self.site_df.loc[self.config['site']]
            region    = site_info['region']
            latitude  = site_info['latitude']
            longitude = site_info['longitude']
            altitude  = site_info['altitude']
                
        else:      
            parameters = ['time','location','latitude','longitude','altitude']
            if not np.all(np.isin(parameters),list(self.config)):
                par_error = 'Please supply '
                for par in parameters:
                    par_error += f'\'{par}\''
                raise Exception(par_error)
            else:
                region    = self.config['region']
                latitude  = self.config['latitude']
                longitude = self.config['longitude']
                altitude  = self.config['altitude']

        self.observer = ephem.Observer()
        self.observer.lat, self.observer.lon, self.observer.elevation = str(latitude), str(longitude), altitude
        self.region    = region
        self.timestamp = self.config['time']
        self.observer.date = datetime.fromtimestamp(self.timestamp)
            
        if 'weather_gen_method' in list(self.config):
            
            self.weather = weathergen.generate(region=self.region,
                                               time=self.timestamp,
                                               method=self.config['weather_gen_method'])
        
            
        if 'pwv' in list(self.config):
            
            self.weather['pwv'] = self.config['pwv'] 
            
class array():
    
    def __init__(self, config=None):
       
        if config==None:
            print('No array config specified, defaulting to a 271-detector hexagonal array.')
            self.config = default_array_config.copy()
        else:
            self.config = config.copy()

        if 'shape' in list(self.config):
            
            validate_config(list(self.config),['shape','fov','n'],name='')
            self.config['offset_x'], self.config['offset_y'] = tools.make_array(self.config['shape'], 
                                                                                self.config['fov'], 
                                                                                self.config['n']) 
            
        self.z = np.radians(self.config['offset_x']) + 1j*np.radians(self.config['offset_y']); self.z -= self.z.mean()
        self.x = np.real(self.z)
        self.y = np.imag(self.z)
        self.n = len(self.z)
        
        # if no band is specified, then use the default setting 
        if not 'nom_bands' in list(self.config):
            self.config['nom_bands'] = default_array_config['nom_bands']
            
        if not 'bandwidths' in list(self.config):
            self.config['bandwidths'] = 1e-1 * self.config['nom_bands']
                        
        #if type(self.config['bands']) in [float,int]:
        #self.bands     = self.config['bands'] * np.ones((self.n))
        #self.band_errs = self.config['band_errs'] * np.ones((self.n))
        self.nom_bands  = self.config['nom_bands'] * np.ones((self.n))
        self.bandwidths = self.config['bandwidths'] * np.ones((self.n))
        
        self.nom_band_list, ui  = np.unique(self.nom_bands,return_index=True)
        self.bandwidth_list     = self.bandwidths[ui[np.argsort(self.nom_band_list)]]
        self.nom_band_list      = np.sort(self.nom_band_list)
        
        self.band_freq_list = np.c_[[mean + width*np.linspace(-.6,.6,121) for mean, width in zip(self.nom_band_list,
                                                                                                self.bandwidth_list)]]
                           
        #
        
        flat_bp = lambda nu, band, width : (np.abs(nu-band) < .5*width).astype(float)
        
        flat_bp = lambda nu, band, width : np.exp(np.log(.5)*(np.abs(nu-band)/(.5*width+1e-16))**8)
        
        self.band_pass_list = np.c_[[flat_bp(freq,mean,width) for freq,mean,width in zip(self.band_freq_list,
                                                                                         self.nom_band_list,
                                                                                         self.bandwidth_list)]]
        
        #self.band_field = np.sort(np.unique(np.r_[[mean + np.sqrt(2*np.log(2))*sigma*np.linspace(-1,1,9) for mean, sigma 
        #                                           in zip(self.bands,self.band_errs)]]))
        
        #self.n_band       = len(self.ubands)
        #self.n_band_field = len(self.band_field)
        #ratio_f = 1.03
        #n_bands = int(np.ceil(np.log(self.band_field.max()/self.band_field.min()) / np.log(ratio_f)))
        #if n_bands < len(self.band_field):
        #    self.band_field = np.geomspace(self.band_field.min(),self.band_field.max(),n_bands)
            
        #self.band_assoc = self.ubands[np.abs(np.subtract.outer(self.ubands,self.band_field)).argmin(axis=0)]
        
        #gaussian_bp = lambda nu, band, band_sig : np.exp(-.5*((nu-band)/(band_sig+1e-16))**8)
        
        
        
        #self.band_weights = flat_bp(self.band_field[None,:],self.nom_bands[:,None],.5*self.bandwidths[:,None])
        #self.band_weights[self.band_weights < 1e-4] = 0
        #self.unit_band_weights = self.band_weights.copy()
        #self.unit_band_weights[self.unit_band_weights==0] = np.nan
        #self.band_weights /= np.nansum(self.band_weights,axis=1)[:,None]
            
        #self.ubands = np.unique(self.bands)
        self.white_noise = self.config['white_noise'] * np.ones((self.n))
            

                
             
class pointing():
    
    def __init__(self, config=None):
        
        if config==None:
            print('No pointing config specified, defaulting to a 10-second zenith stare at 20 Hz.')
            self.config = default_pointing_config
        else:
            self.config = config.copy()
            
        if 'scan_type' in list(self.config):
            
            self.duration = self.config['duration']
            self.dt   = 1 / self.config['samp_freq']
            self.time = np.arange(0, self.duration, self.dt)
            self.nt   = len(self.time)
            self.f_   = np.fft.fftfreq(self.nt,self.dt)
            
            self.center_azim, self.center_elev = np.radians(self.config['center_azim']), np.radians(self.config['center_elev'])
            
            if self.config['scan_type']=='CES':
                
                self.scan_freq  = self.config['az_speed'] / (4*self.config['az_throw']+1e-16)
                self.focal_azim = (self.center_azim + np.radians(self.config['az_throw'])*sp.signal.sawtooth(np.pi/2 + 2*np.pi*self.scan_freq*self.time,width=.5)) % (2*np.pi)
                self.focal_elev = self.center_elev + np.zeros(self.nt)
                
            if self.config['scan_type']=='lissajous_box':

                focal_x = np.radians(self.config['x_throw']) * np.sin(2*np.pi*self.time/self.config['x_period'])
                focal_y = np.radians(self.config['y_throw']) * np.sin(2*np.pi*self.time/self.config['y_period'])
                
                self.focal_azim, self.focal_elev = tools.from_xy(focal_x,focal_y,self.center_azim,self.center_elev)
                
            if self.config['scan_type']=='lissajous_daisy':

                focal_r = np.radians(self.config['throw']) * np.sin(2*np.pi*self.time/self.config['r_period'])
                focal_p = 2*np.pi*self.time/self.config['p_period']
    
                focal_x, focal_y = focal_r * np.cos(focal_p), focal_r * np.sin(focal_p)
                
                self.focal_azim, self.focal_elev = tools.from_xy(focal_x,focal_y,self.center_azim,self.center_elev)
            
        else:
            
            self.focal_azim = self.config['focal_azim']
            self.focal_elev = self.config['focal_elev']
            self.time       = self.config['time']
            self.duration   = self.time.max() - self.time.min()
            self.dt = np.gradient(self.time).mean()
            self.nt = len(self.time)
            self.f_ = np.fft.fftfreq(self.nt,self.dt)
        
        

class beams():
    
     def __init__(self, config=None):
        
        if config==None:
            print('No beam config specified, defaulting to a 5-meter diffraction-limited beam.')
            self.config = default_beams_config
        else:
            self.config = config.copy()

        for arg in list(default_beams_config):
            if not arg in list(self.config):
                self.config[arg] = default_beams_config[arg]
                
        if self.config['optical_type'] == 'diff_lim':
            
            self.aperture = self.config['primary_size']
            self.min_beam_res = self.config['min_beam_res']
            #self.n_bf     = int(1.5*np.ceil(self.min_beam_res))
            
            # we define the waist as the FWHM of the beam cross-section

            #gauss_half_waist = lambda z, w_0, f : .5 * w_0 * np.sqrt(1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(.5 * w_0)))
            #sharp_half_waist = lambda z, w_0, f : .5 * np.maximum(w_0,1.27324 * 2.998e8 * z / (w_0 * f))
            
            
            
            self.get_waist = lambda z, w_0, f : np.maximum(w_0,1.27324 * 2.998e8 * z / (w_0 * f))
            
        if self.config['beam_model'] == 'top_hat':
            self.get_window = lambda r, hwhm : np.exp(np.log(.5)*(r/hwhm)**8)
        if self.config['beam_model'] == 'gaussian':
            self.get_window = lambda r, hwhm : np.exp(np.log(.5)*(r/hwhm)**2)

#atmosphere()
