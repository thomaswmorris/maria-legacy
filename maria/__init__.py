import time
import scipy as sp
import scipy.cluster
import numpy as np
import numpy.linalg as la

import warnings


from tqdm import tqdm
from . import tools
from . import objects

#import tools
#import objects

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#exec(open('mpl_defs').read())



default_misc_config = {'fov_tolerance' : 1e-2, # proportion larger of the FOV to make the generated atmosphere 
                          'n_fov_edge' : 16, 
             'n_center_azimuth_sample' : 16, 
              'n_sidereal_time_sample' : 16, 
                        'n_sample_max' : 10000}


class model():
    
    def __init__(self, site_config=None,
                       array_config=None, 
                       beams_config=None, 
                       pointing_config=None, 
                       atmosphere_config=None,
                       misc_config=default_misc_config, verbose=False, show_warnings=True):
        
        self.site       = objects.site(config=site_config)
        self.array      = objects.array(config=array_config)
        self.beams      = objects.beams(config=beams_config)
        self.pointing   = objects.pointing(config=pointing_config)
        self.atmosphere = objects.atmosphere(config=atmosphere_config)
        
        self.misc_config = misc_config.copy()
        
        
        ### ==================================================================
        ### We've sorted a lot of parameters into interactive classes. But we need to combine them to find some model parameters.
        
        
        # Here we compute parameters related to the beam waists for each layer. This means different things for different beam models.
        
        
        self.beam_waists = self.beams.get_waist(self.atmosphere.depths[:,None],
                                                self.beams.aperture,
                                                self.array.nom_band_list[None,:])
        
        self.ang_waists = self.beam_waists / self.atmosphere.depths[:,None]
        
        self.min_ang_res = self.ang_waists / self.beams.min_beam_res # np.minimum(np.radians(1), )    
        
        # Here we make an imaginary array that bounds the true array, which helps to determine the geometry of the atmosphere to simulate. 
        self.array.edge_r = (1+misc_config['fov_tolerance']) * (np.abs(self.array.z).max() + self.ang_waists.max(axis=1)/2)
        self.array.edge_z = self.array.edge_r[:,None] * np.exp(1j*np.linspace(0,2*np.pi,misc_config['n_fov_edge']+1)[1:])[None,:]
        self.array.edge_x, self.array.edge_y = np.real(self.array.edge_z), np.imag(self.array.edge_z)
        
        self.atmosphere.turb_nu = 5/6
        
        # this defines the relative strength of pwv fluctuations
        self.atmosphere.heights = self.atmosphere.depths * np.sin(self.pointing.center_elev.mean())
        self.atmosphere.wvmd = np.interp(self.atmosphere.heights,self.site.weather['height']-self.site.observer.elevation,self.site.weather['water_density'])
        self.atmosphere.temp = np.interp(self.atmosphere.heights,self.site.weather['height']-self.site.observer.elevation,self.site.weather['temperature'])
        self.var_scaling = np.square(self.atmosphere.wvmd * self.atmosphere.temp)
        self.rel_scaling = np.sqrt(self.var_scaling/self.var_scaling.sum())
        self.lay_scaling = self.site.weather['pwv'] * self.atmosphere.config['rel_atm_rms'] * self.rel_scaling * self.atmosphere.thicks / self.atmosphere.thicks.sum()
        
        ### ==================================================================
        ### Here we compute time-ordered pointing angles and velocities. 
        self.pointing.azim, self.pointing.elev = tools.from_xy(self.array.x[:,None], 
                                                               self.array.y[:,None], 
                                                               self.pointing.focal_azim[None,:], 
                                                               self.pointing.focal_elev[None,:])

        self.pointing.azim_motion = (np.gradient(np.sin(self.pointing.focal_azim)) * np.cos(self.pointing.focal_azim) 
                                    -np.gradient(np.cos(self.pointing.focal_azim)) * np.sin(self.pointing.focal_azim)) / np.gradient(self.pointing.time) 
        
        self.pointing.elev_motion = (np.gradient(np.sin(self.pointing.focal_elev)) * np.cos(self.pointing.focal_elev) 
                                    -np.gradient(np.cos(self.pointing.focal_elev)) * np.sin(self.pointing.focal_elev)) / np.gradient(self.pointing.time) 
        
        # Here we compute the focal angular velocities
        self.atmosphere.wind_east  = np.interp(self.atmosphere.depths,self.site.weather['height']-self.site.observer.elevation,self.site.weather['wind_east']) 
        self.atmosphere.wind_north = np.interp(self.atmosphere.depths,self.site.weather['height']-self.site.observer.elevation,self.site.weather['wind_north']) 
        self.atmosphere.wind_speed = np.abs(self.atmosphere.wind_north+1j*self.atmosphere.wind_east)
        self.atmosphere.wind_bear  = np.angle(self.atmosphere.wind_north+1j*self.atmosphere.wind_east)
    
        self.pointing.omega_x = (self.atmosphere.wind_east[:,None] * np.cos(self.pointing.focal_azim[None,:]) \
                              - self.atmosphere.wind_north[:,None] * np.sin(self.pointing.focal_azim[None,:])) / self.atmosphere.depths[:,None] \
                           + self.pointing.azim_motion[None,:] * np.cos(self.pointing.focal_elev[None,:]) 
                           
        self.pointing.omega_y = -(self.atmosphere.wind_east[:,None] * np.sin(self.pointing.focal_azim[None,:]) \
                               - self.atmosphere.wind_north[:,None] * np.cos(self.pointing.focal_azim[None,:])) / self.atmosphere.depths[:,None] * np.sin(self.pointing.focal_elev[None,:]) \
                           + self.pointing.elev_motion[None,:]

        self.pointing.omega_z = self.pointing.omega_x + 1j*self.pointing.omega_y
                           
        
        # Here we compute the time-ordered focal angular positions
        
        self.pointing.focal_theta_z = np.cumsum(self.pointing.omega_z * np.gradient(self.pointing.time)[None,:],axis=-1)
        
        self.pointing.theta_z  = self.array.z[None,:,None] + self.pointing.focal_theta_z[:,None,:]
        #self.pointing.theta_z -= self.pointing.theta_z.mean()
        self.pointing.theta_x, self.pointing.theta_y = np.real(self.pointing.theta_z), np.imag(self.pointing.theta_z)
        
        ### These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer. 
        self.para, self.orth, self.X, self.Y, self.P, self.O = [], [], [], [], [], []
        self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = [], [], [], [], []
        
        self.pointing.zop = np.zeros((self.pointing.theta_z.shape),dtype=complex)
        self.pointing.p   = np.zeros((self.pointing.theta_z.shape))
        self.pointing.o   = np.zeros((self.pointing.theta_z.shape))
        
        self.MARA = []
        self.atmosphere.outer_scale, self.atmosphere.turb_nu = self.atmosphere.config['outer_scale'], 5/6
        self.ang_outer_scale = self.atmosphere.outer_scale / self.atmosphere.depths
        
        aam_weight = np.square(self.atmosphere.depths * self.lay_scaling)[:,None]
        
        self.atmosphere.aam  = np.sum(aam_weight*self.pointing.omega_z,axis=0) / np.sum(aam_weight*np.square(self.pointing.omega_z),axis=0)
        self.atmosphere.aam /= np.square(np.abs(self.atmosphere.aam))
        
        self.pointing.theta_edge_z = []
        
        for i_l, depth in enumerate(self.atmosphere.depths):
                        
            # an efficient way to compute the minimal observing area that we need to generate
            
            try:
                edge_hull = sp.spatial.ConvexHull(points=np.vstack([np.real(self.pointing.focal_theta_z[i_l]).ravel(),
                                                                    np.imag(self.pointing.focal_theta_z[i_l]).ravel()]).T)
                edge_hull_z  = self.pointing.focal_theta_z[i_l].ravel()[edge_hull.vertices]
            except:
                edge_hull_z = self.pointing.focal_theta_z[i_l][np.array([0,-1])]
            
            
            theta_edge_z = self.array.edge_z[i_l][:,None] + edge_hull_z[None,:]
            
            self.pointing.theta_edge_z.append(theta_edge_z)

            self.MARA.append(tools.get_MARA(theta_edge_z.ravel()))
            RZ = theta_edge_z * np.exp(1j*self.MARA[-1])
            
            para_min, para_max = np.real(RZ).min(), np.real(RZ).max()
            orth_min, orth_max = np.imag(RZ).min(), np.imag(RZ).max()
            
            para_center, orth_center = (para_min + para_max)/2, (orth_min + orth_max)/2
            para_radius, orth_radius = (para_max - para_min)/2, (orth_max - orth_min)/2
    
            n_orth_min = 64
            n_orth_max = 1024

            lay_ang_res = np.minimum(self.min_ang_res[i_l].min(), 2 * orth_radius / (n_orth_min - 1))
            lay_ang_res = np.maximum(lay_ang_res, 2 * orth_radius / (n_orth_max - 1))
       
            
            self.lay_ang_res.append(lay_ang_res)
            
            para_ = para_center + np.arange(-para_radius,para_radius+.5*lay_ang_res,lay_ang_res)
            orth_ = orth_center + np.arange(-orth_radius,orth_radius+.5*lay_ang_res,lay_ang_res)
            
            self.PARA_SPACING = np.gradient(para_).mean()
            self.para.append(para_), self.orth.append(orth_)
            self.n_para.append(len(para_)), self.n_orth.append(len(orth_))
        
            ORTH_,PARA_ = np.meshgrid(orth_,para_)
            
            self.genz.append(np.exp(-1j*self.MARA[-1]) * (PARA_[0] + 1j*ORTH_[0] - self.PARA_SPACING) )
            layer_ZOP = np.exp(-1j*self.MARA[-1]) * (PARA_ + 1j*ORTH_) 
            
            self.X.append(np.real(layer_ZOP)), self.Y.append(np.imag(layer_ZOP))
            self.O.append(ORTH_), self.P.append(PARA_)
            
            
            
            self.pointing.zop[i_l] = self.pointing.theta_z[i_l] * np.exp(1j*self.MARA[-1]) 
            self.pointing.p[i_l], self.pointing.o[i_l] = np.real(self.pointing.zop[i_l]), np.imag(self.pointing.zop[i_l])

            cov_args = (1,1)
            
            para_i, orth_i = [],[]
            for ii,i in enumerate(np.r_[0,2**np.arange(np.ceil(np.log(self.n_para[-1])/np.log(2))),self.n_para[-1]-1]):
                
                #if i * self.ang_res[i_l] > 2 * self.ang_outer_scale[i_l]:
                #    continue
                
                #orth_i.append(np.unique(np.linspace(0,self.n_orth[-1]-1,int(np.maximum(self.n_orth[-1]/(i+1),16))).astype(int)))
                orth_i.append(np.unique(np.linspace(0,self.n_orth[-1]-1,int(np.maximum(self.n_orth[-1]/(4**ii),4))).astype(int)))
                para_i.append(np.repeat(i,len(orth_i[-1])).astype(int))
                
            self.AR_samples.append((np.concatenate(para_i),np.concatenate(orth_i)))
            
            n_cm = len(self.AR_samples[-1][0])
            
            if n_cm > 5000 and show_warnings:
                
                warning_message = f'A very large covariance matrix for layer {i_l+1} (n_side = {n_cm})'
                warnings.warn(warning_message)
            
            
            
        if verbose:
            print('\n # | depth (m) | beam (m) | beam (\') | sim (m) | sim (\') | rms (mg/m2) | n_cov | orth | para | h2o (mg/m3) | temp (K) | ws (m/s) | wb (deg) |')
            
            for i_l, depth in enumerate(self.atmosphere.depths):
                
                row_string  = f'{i_l+1:2} | {depth:9.01f} | {self.beam_waists[i_l].min():8.02f} | {60*np.degrees(self.ang_waists[i_l].min()):8.02f} | '
                row_string += f'{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | '
                row_string += f'{1e3*self.lay_scaling[i_l]:11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | '
                row_string += f'{self.n_para[i_l]:4} | {1e3*self.atmosphere.wvmd[i_l]:11.02f} | {self.atmosphere.temp[i_l]:8.02f} | '
                row_string += f'{self.atmosphere.wind_speed[i_l]:8.02f} | {np.degrees(self.atmosphere.wind_bear[i_l]+np.pi):8.02f} |'
                print(row_string)
                
        
        self.prec, self.csam, self.cgen, self.A, self.B = [], [], [], [], []
        
        with tqdm(total=len(self.atmosphere.depths),desc='Computing weights') as prog:
            for i_l, (depth, LX, LY, AR, GZ) in enumerate(zip(self.atmosphere.depths,self.X,self.Y,self.AR_samples,self.genz)):
                
                cov_args  = (self.atmosphere.outer_scale / depth, self.atmosphere.turb_nu)
                
                self.prec.append(la.inv(tools.make_2d_covariance_matrix(self.atmosphere.matern,cov_args,LX[AR],LY[AR])))

                self.cgen.append(tools.make_2d_covariance_matrix(self.atmosphere.matern,cov_args,np.real(GZ),np.imag(GZ)))
                
                self.csam.append(tools.make_2d_covariance_matrix(self.atmosphere.matern,cov_args,np.real(GZ),np.imag(GZ),LX[AR],LY[AR],auto=False)) 
                
                self.A.append(np.matmul(self.csam[i_l],self.prec[i_l])) 
                self.B.append(tools.msqrt(self.cgen[i_l]-np.matmul(self.A[i_l],self.csam[i_l].T)))
                
                prog.update(1)
        
    def atmosphere_timestep(self,i): # iterate the i-th layer of atmosphere by one step
        
        self.vals[i] = np.r_[(np.matmul(self.A[i],self.vals[i][self.AR_samples[i]])
                            + np.matmul(self.B[i],np.random.standard_normal(self.B[i].shape[0])))[None,:],self.vals[i][:-1]]

    def generate_atmosphere(self,blurred=False):

        self.vals = [np.zeros(lx.shape) for lx in self.X]
        n_init_   = [n_para for n_para in self.n_para]
        n_ts_     = [n_para for n_para in self.n_para]
        tot_n_init, tot_n_ts = np.sum(n_init_), np.sum(n_ts_)
        #self.gen_data = [np.zeros((n_ts,v.shape[1])) for n_ts,v in zip(n_ts_,self.lay_v_)]

        with tqdm(total=tot_n_init,desc='Generating layers') as prog:
            for i, n_init in enumerate(n_init_):
                for i_init in range(n_init):
                    
                    self.atmosphere_timestep(i)
                    
                    prog.update(1)
                
    
    def sim(self, do_atmosphere=True, 
                  units='mK_CMB', 
                  do_noise=True,
                  split_layers=False,
                  split_bands=False):
        
        self.sim_start = time.time()
        self.generate_atmosphere()
        
        #print(self.array.band_weights.shape)
        #temp_data = np.zeros((len(self.atmosphere.depths),self.array.n,self.pointing.nt))
        
        self.epwv = self.site.weather['pwv'] + np.zeros((self.array.n,self.pointing.nt))
        
        self.n_bf, self.beam_filters, self.beam_filter_sides = [], [], []
        
        with tqdm(total=len(self.atmosphere.depths) + len(self.array.nom_band_list),
                  desc='Sampling atmosphere') as prog:
            
            
            for i_l, depth in enumerate(self.atmosphere.depths): 
                
                waist_samples, which_sample = tools.smallest_max_error_sample(self.ang_waists[i_l],max_error=1e-1)
                
                wv_data = np.zeros((self.array.n,self.pointing.nt))
                
                for i_w, w in enumerate(waist_samples):
                #for i_ba, nom_band in enumerate(self.array.nom_band_list):
                    
                    # band-waist mask : which detectors observe bands that are most effectively modeled by this waist?
                    bm = np.isin(self.array.nom_bands,self.array.nom_band_list[which_sample == i_w])
                    
                    # filter the angular atmospheric emission, to account for beams
                    self.n_bf.append(int(np.ceil(.6 * w / self.lay_ang_res[i_l])))
                    self.beam_filter_sides.append(self.lay_ang_res[i_l] * np.arange(-self.n_bf[-1],self.n_bf[-1]+1))
                    self.beam_filters.append(tools.make_beam_filter(self.beam_filter_sides[-1],self.beams.get_window,[w/2]))
                
                    
                    filtered_vals = sp.signal.convolve2d(self.vals[i_l], self.beam_filters[-1], boundary='symm', mode='same')
                    #sigma = .5 * w / self.lay_ang_res[i_l]
                    #print(sigma)
                    #filtered_vals = sp.ndimage.gaussian_filter(self.vals[i_l], sigma=sigma)
                    
                    FRGI = sp.interpolate.RegularGridInterpolator((self.para[i_l],self.orth[i_l]), self.lay_scaling[i_l] * filtered_vals)
                    wv_data[bm] = FRGI((self.pointing.p[i_l][bm],self.pointing.o[i_l][bm]))
                   
                    
                self.epwv += wv_data
                prog.update(1)
                    
            #self.los_wv /= np.sin(self.pointing.elev)
            atm_temp_data = np.zeros((self.array.n,self.pointing.nt))
            
            for i_ba, (nom_band,f_ps,ps) in enumerate(zip(self.array.nom_band_list,
                                                          self.array.band_freq_list,
                                                          self.array.band_pass_list)):
                
                bm = self.array.nom_bands == nom_band
                
                min_elev, max_elev = self.pointing.elev[bm].min(), self.pointing.elev[bm].max()
                min_epwv, max_epwv = self.epwv[bm].min(), self.epwv[bm].max()
                
                i_min_elev = np.where(self.atmosphere.spectra_dict['elev']<=min_elev)[0][-1]
                i_max_elev = np.where(self.atmosphere.spectra_dict['elev']>=max_elev)[0][0] + 1
                i_min_epwv = np.where(self.atmosphere.spectra_dict['epwv']<=min_epwv)[0][-1]
                i_max_epwv = np.where(self.atmosphere.spectra_dict['epwv']>=max_epwv)[0][0] + 1
                
                gridded_temp = 1e3 * tools.get_brightness_temperature(f_ps, ps, self.atmosphere.spectra_dict['freq'], self.atmosphere.spectra_dict['temp'])
                
                TRGI = sp.interpolate.RegularGridInterpolator((self.atmosphere.spectra_dict['elev'][i_min_elev:i_max_elev],
                                                               self.atmosphere.spectra_dict['epwv'][i_min_epwv:i_max_epwv]),
                                                               gridded_temp[i_min_elev:i_max_elev,i_min_epwv:i_max_epwv])
                
                #print(self.atmosphere.spectra_dict['elev'][i_min_elev],self.atmosphere.spectra_dict['elev'][i_max_elev])
                #print(self.pointing.elev[bm].min(),self.pointing.elev[bm].max())
                
                #print(self.atmosphere.spectra_dict['epwv'][i_min_epwv:i_max_epwv].min(),self.atmosphere.spectra_dict['epwv'][i_min_epwv:i_max_epwv][i_min_elev:i_max_elev].max())
                #print(self.epwv[bm].min(),self.epwv[bm].max())
                
                atm_temp_data[bm] = TRGI((self.pointing.elev[bm],self.epwv[bm]))
                
                prog.update(1)
            
            #for iband, band in enumerate(self.array.band_field):
                
                #bm = self.array.band_weights[:,iband] > 0
                #ib_assoc = np.where(self.array.band_assoc[iband]==self.array.ubands)[0]
                #atm_temp_data[bm] += self.array.band_weights[bm,iband][:,None] * self.temp_interpolators[band](self.los_wv[bm])
                
                #prog.update(1)
    
        #temp_data = band_temp_data if split_bands else band_temp_data.sum(axis=0)            

        self.sim_end = time.time()
        
        print(f'\ndone!\nsim time:  {self.pointing.duration:8.01f}s\nwall time: {self.sim_end-self.sim_start:8.01f}s')
        
        self.atm_data   = sp.ndimage.gaussian_filter1d(atm_temp_data,sigma=1,axis=-1)
        
        noise_data = 0
        if do_noise == True:
        
            noise_data = self.array.white_noise.reshape(self.array.n,1) / self.pointing.dt * np.random.standard_normal(atm_temp_data.shape)
        
        self.noise_data = noise_data
        self.tot_data   = self.atm_data + self.noise_data

        #return atm_temp_data + noise_data
    
    
def get_pair_lags(data_,              # time-ordered data, shape (nd,nt)
                  offset_x,           # horizontal detector offset (nd,)
                  offset_y,           # vertical detector offset (nd,)
                  time_,              # time field (nt,)
                  azim_,              # azimuth field (nt,)
                  elev_,              # elevation field (nt,)
                  n_clusters=None,    # how many clusters to consolidate detectors into 
                  sub_scan_durs=[]):  # how many clusters to consolidate detectors into 
                  
    high_pass = lambda data, c, fs, order=3 : sp.signal.filtfilt(*sp.signal.butter(order,2*c/fs,btype='highpass'),data)

    dt         = np.median(np.gradient(time_))
    sub_splits = tools.get_sub_splits(time_,np.gradient(azim_)/np.gradient(time_))
    
    print(sub_splits)

    if n_clusters is None: 
        clust_data = data_
        clust_x, clust_y = offset_x, offset_y
        
    else:
        init_clust_x, init_clust_y = tools.make_array('flower', np.maximum(np.ptp(offset_x),np.ptp(offset_y)), n_clusters) 
        init_clust_x += offset_x.mean(); init_clust_y += offset_y.mean()
        cluster_id = np.abs(np.subtract.outer(init_clust_x+1j*init_clust_y,offset_x+1j*offset_y)).argmin(axis=0)
        
        clust_data = np.concatenate([data_[cluster_id==i].mean(axis=0)[None,:] for i in np.sort(np.unique(cluster_id))],axis=0)
        clust_x    = np.array([offset_x[cluster_id==i].mean(axis=0) for i in np.sort(np.unique(cluster_id))])
        clust_y    = np.array([offset_y[cluster_id==i].mean(axis=0) for i in np.sort(np.unique(cluster_id))])
        
    filtered_clust_data = high_pass(clust_data,2e-1,1/dt,order=3)

    # use Fourier methods to compute the pair-lag, for each sub-split and each cluster pair
    pair_lags = np.zeros((sub_splits.shape[0], clust_data.shape[0], clust_data.shape[0]))
    for i_spl,(s,e) in enumerate(sub_splits):
        
        ft_sub_data = np.fft.fft(filtered_clust_data[:,s:e]*np.hanning(e-s), axis=-1)
        
        for i_det in range(clust_data.shape[0]):
            for j_det in range(i_det):
                
                pair_lags[i_spl,i_det,j_det] = np.fft.ifft(ft_sub_data[i_det]*np.conj(ft_sub_data[j_det])).argmax().astype(float)
                
        pair_lags[i_spl][pair_lags[i_spl] > (e-s)/2] -= e-s
        pair_lags[i_spl] += - pair_lags[i_spl].T
        pair_lags[i_spl] *= dt
        
    return pair_lags, clust_x, clust_y

        
def fit_pair_lags(pair_lags, 
                  clust_x, 
                  clust_y,
                  max_lag,       # degrees per second
                  max_vel,       # degrees per second
                  weights=None):   
                  

    lag_flat = lambda dz, vx, vy : - (vx*np.real(dz) + vy*np.imag(dz)) / np.square(np.abs(vx+1j*vy) + 1e-16)
    
    
    
    clust_outer_z = np.subtract.outer(clust_x,clust_x) + 1j*np.subtract.outer(clust_y,clust_y)
        
    vel_pars = np.zeros((pair_lags.shape[0],2))
    bounds   = [[-max_vel,-max_vel],[max_vel,max_vel]]
    
    for i_spl in range(pair_lags.shape[0]):
        
        use = (np.abs(pair_lags[i_spl]) < max_lag) & (np.abs(pair_lags[i_spl]) > 0) 
        
        if not use.sum() > 4:
            continue
        
        max_sep_z = np.mean(clust_outer_z[use][pair_lags[i_spl,use]==pair_lags[i_spl,use].max()])
        vz0 = - max_sep_z / pair_lags[i_spl,use].max()
        vx0, vy0 = np.real(vz0), np.imag(vz0)
        
        #print(vx0,vy0)
        #print(lag_flat(clust_outer_z,vx0,vy0))
        
        pars,cpars = sp.optimize.curve_fit(lag_flat,
                                           clust_outer_z[use],
                                           pair_lags[i_spl,use],
                                           p0=[vx0,vy0],
                                           bounds=bounds,
                                           maxfev=10000)
        vel_pars[i_spl] = pars
        
    return vel_pars
    
        
    
        

    
    
    
    
    pass
    