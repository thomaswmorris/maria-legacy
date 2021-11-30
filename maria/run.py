# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from datetime import datetime
from datetime import timezone
import maria

        
atmosphere_config = {'n_layers'   : 32,         # how many layers to simulate, based on the integrated atmospheric model 
                    'min_depth'   : 100,      # the height of the first layer 
                    'max_depth'   : 3000,      # 
                    'rel_atm_rms' : 5e-1,  
                    'outer_scale' : 500}


pointing_config = {'scan_type' : 'lissajous_box',
                    'duration' : 300,'samp_freq' : 50,
                 'center_azim' : -45, 'center_elev' : 45, 
                     'x_throw' : 5, 'x_period' : 21,
                     'y_throw' : 5, 'y_period' : 29}

pointing_config = {'scan_type' : 'CES',
                    'duration' : 60, 'samp_freq' : 20,
                 'center_azim' : 55, 'center_elev' : 45, 
                    'az_throw' : 15, 'az_speed' : 1}


n_per = 128

nom_bands = np.r_[3.8e10*np.ones(n_per),
                  9.6e10*np.ones(n_per),
                  2.2e11*np.ones(n_per),
                  1.5e11*np.ones(n_per)][n_per:]


#bands = 1.5e11*np.ones(240)
#np.random.shuffle(bands)
#bands = 1.5e11 * np.ones(n_per)

bandwidths = 2.5e-1 * nom_bands

wns = 1e-1 #* np.sqrt(nom_bands / nom_bands.min()) 

array_config = {'shape' : 'flower',
                    'n' : len(nom_bands),      
                  'fov' : 1.,
            'nom_bands' : nom_bands,
           'bandwidths' : bandwidths,
          'white_noise' : wns}   



beams_config = {'optical_type' : 'diff_lim',
                'primary_size' : 5,
                  'beam_model' : 'top_hat',
                'min_beam_res' : 1 }    

site_config = {'site' : 'ACT',
               'time' : datetime.now(timezone.utc).timestamp(),
 'weather_gen_method' : 'mean',
                'pwv' : 1, 
             'region' : 'atacama' }

heights = np.linspace(0,10000,100)


import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['figure.dpi'] = 256


def equalize(ax):
    xls,yls = ax.get_xlim(),ax.get_ylim()
    x0,y0 = np.mean(xls), np.mean(yls)
    r = np.maximum(-np.subtract(*xls),-np.subtract(*yls))/2
    ax.set_xlim(x0-r,x0+r); ax.set_ylim(y0-r,y0+r)
    return ax

new = True
sim = True




tm = maria.model(atmosphere_config=atmosphere_config,
           pointing_config=pointing_config,
           beams_config=beams_config,
           array_config=array_config,
           site_config=site_config,
           verbose=True)
        
tm.sim()

data = tm.tot_data

pair_lags, clust_x, clust_y = maria.get_pair_lags(data,
                                 tm.array.x,
                                 tm.array.y,
                                 tm.pointing.time,
                                 tm.pointing.focal_azim,
                                 tm.pointing.focal_elev,
                                 n_clusters=6,
                                 sub_scan_durs=[10])

lag_flat = lambda dz, vx, vy : - (vx*np.real(dz) + vy*np.imag(dz)) / np.square(np.abs(vx+1j*vy) + 1e-16)

max_vel = 2
max_lag = 2

vel_pars = maria.fit_pair_lags(pair_lags, clust_x, clust_y, max_lag=max_lag, max_vel=max_vel)

fig,axes = plt.subplots(1,2,figsize=(16,8))

OX, OY = np.subtract.outer(clust_x, clust_x), np.subtract.outer(clust_y, clust_y)
OZ     = OX + 1j*OY
OA, OR = np.angle(OZ), np.degrees(np.abs(OZ) + 1e-16)

a_ = np.linspace(-np.pi,np.pi,64)

for i_spl in range(pair_lags.shape[0]):

    axes[0].scatter(OA, pair_lags[i_spl]/OR)
    
    axes[0].plot(a_,lag_flat(np.exp(1j*a_),*vel_pars[i_spl]) / np.degrees(1),label=f'{i_spl}')
    
    axes[1].scatter(*np.degrees(vel_pars[i_spl]))   
    
axes[0].legend()
    
axes[0].set_ylim(-max_lag,max_lag)

axes[1].plot([-max_vel,max_vel],[0,0],color='k')
axes[1].plot([0,0],[-max_vel,max_vel],color='k')
axes[1].set_xlim(-max_vel,max_vel), axes[1].set_ylim(-max_vel,max_vel)
        
#assert False

exec(open('/Users/thomas/Desktop/atmosphere/mpl_defs').read())

do_clouds = True

if do_clouds:
    
    i = 0

    fig,ax = plt.subplots(1,1,figsize=(8,8))

    ax.pcolormesh(np.degrees(tm.X[i]),
                  np.degrees(tm.Y[i]),
                  tm.vals[i],shading='none',cmap='RdBu_r')
    
    ax.scatter(np.degrees(np.real(tm.pointing.theta_edge_z[i]).T),
               np.degrees(np.imag(tm.pointing.theta_edge_z[i]).T),s=1e-1,c='k')
    
    ax.plot(np.degrees(np.real(tm.pointing.focal_theta_z[i])),
            np.degrees(np.imag(tm.pointing.focal_theta_z[i])),c='k')
    
    equalize(ax)

data_fig, data_axes = plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharex=True)
spec_fig, spec_axes = plt.subplots(1,2,figsize=(10,6),constrained_layout=True,sharey=True)
band_fig, band_axes = plt.subplots(1,2,figsize=(12,6),constrained_layout=True)

nf    = 256
freq  = np.fft.fftfreq(tm.pointing.nt,tm.pointing.dt)
fmids = np.geomspace(2/tm.pointing.duration,freq.max(),nf)
rfreq = np.exp(np.gradient(np.log(fmids))).mean()
fbins = np.append(fmids/np.sqrt(rfreq),fmids[-1]*np.sqrt(rfreq)) 
pt    = np.linspace(0,2*np.pi,16)


from matplotlib.patches import Patch
from matplotlib.collections import EllipseCollection

handles = []
for iband, band in enumerate(tm.array.nom_band_list[::-1]):
    
    color = mpl.cm.get_cmap('plasma')(.9*(len(tm.array.nom_band_list)-iband-1)/(1e-6+len(tm.array.nom_band_list)-1))
    handles.append(Patch(label=f'{np.round(band*1e-9,1)} GHz',color=color))
    bm = tm.array.nom_bands==band
    hwhm = 1.22 * (2.998e8 / band) / tm.beams.aperture / 2
    
    # PLOT DATA 
    for idet, det in enumerate(np.random.choice(data[bm].shape[0],4,replace=False)):

        data_axes[0].plot(tm.pointing.time,tm.epwv[bm][det],color=color)
        data_axes[1].plot(tm.pointing.time,1e-3*data[bm][det],color=color)
        data_axes[2].plot(tm.pointing.time,data[bm][det]-data[bm][det].mean(axis=0),color=color)
        
    # PLOT SPECTRA
    ps    = np.square(np.abs(np.fft.fft(data[bm] * np.hanning(data[bm].shape[-1])[None,:],axis=-1))) 
    ps   *= (2*tm.pointing.dt/np.hanning(data[bm].shape[-1]).sum())
    ncps  = np.square(np.abs(np.fft.fft((data[bm]-data[bm].mean(axis=0)) * np.hanning(data[bm].shape[-1])[None,:],axis=-1)))
    ncps *= (2*tm.pointing.dt/np.hanning(data[bm].shape[-1]).sum())
    mps   = ps.mean(axis=0); bmps = sp.stats.binned_statistic(freq,mps,bins=fbins,statistic='mean')[0]
    ncmps = ncps.mean(axis=0); ncbmps = sp.stats.binned_statistic(freq,ncmps,bins=fbins,statistic='mean')[0]
    nn    = ~np.isnan(bmps)
    spec_axes[0].plot(fmids[nn],bmps[nn],color=color)
    spec_axes[1].plot(fmids[nn],ncbmps[nn],color=color)
    
    # PLOT BANDS 
    band_axes[0].plot(np.degrees(tm.array.x[bm][None,:] + hwhm*np.cos(pt[:,None])),
                      np.degrees(tm.array.y[bm][None,:] + hwhm*np.sin(pt[:,None])),
                      lw=1e0,color=color)
    
    ec = EllipseCollection(np.degrees(2*hwhm)*np.ones(bm.sum()), 
                           np.degrees(2*hwhm)*np.ones(bm.sum()),
                           np.zeros(bm.sum()), units='x', 
                           offsets=np.degrees(np.column_stack([tm.array.x[bm],tm.array.y[bm]])),
                           transOffset=band_axes[0].transData,
                           color=color,alpha=.5)

    band_axes[0].add_collection(ec)
    
    band_axes[0].set_xlabel(r'$\theta_x$ (degrees)'); band_axes[0].set_ylabel(r'$\theta_y$ (degrees)')

    i_epwv    = np.argmin(np.abs(tm.atmosphere.spectra_dict['epwv'] - tm.site.weather['pwv']))
    i_elev    = np.argmin(np.abs(tm.atmosphere.spectra_dict['elev'] - tm.pointing.elev.mean()))
    spectrum  = tm.array.band_pass_list[iband] * np.interp(tm.array.band_freq_list[iband], 
                                                           tm.atmosphere.spectra_dict['freq'],
                                                           tm.atmosphere.spectra_dict['temp'][i_elev,i_epwv])
    
    band_axes[1].plot(1e-9*tm.array.band_freq_list[iband], spectrum, color=color, lw=5, alpha=.5)
    #band_axes[1].scatter(1e-9*tm.array.band_freq_list[iband], spectrum, color=color, s=16)
    
    
data_axes[2].set_xlabel('$t$ (s)')
data_axes[0].set_ylabel(r'$P_\mathrm{eff}(t)$ (mm)')
data_axes[1].set_ylabel(r'$T_\mathrm{atm}(t)$ (K$_\mathrm{CMB}$)')
data_axes[2].set_ylabel(r'$\Delta T_\mathrm{atm}(t)$ (mK$_\mathrm{CMB}$)')

spec_axes[0].plot(fmids[nn],(bmps[nn][-1]/1e0**(-8/3))*fmids[nn]**(-8/3),color='k')
spec_axes[1].plot(fmids[nn],(ncbmps[nn][-1]/1e0**(-8/3))*fmids[nn]**(-8/3),color='k')

spec_axes[0].loglog()
spec_axes[1].loglog()

band_axes[1].plot(1e-9*tm.atmosphere.spectra_dict['freq'], tm.atmosphere.spectra_dict['temp'][i_elev,i_epwv],color='k')
band_axes[1].set_xlabel(r'$\nu$ (GHz)'); band_axes[1].set_ylabel('Temperature (K)')
band_axes[1].grid(b=False)
band_axes[1].set_xscale('log')
band_axes[1].set_ylim(0,60)

band_ticks = np.array([30,40,60,100,150,200,300,500,1000])

band_axes[1].set_xticks(band_ticks)

band_axes[1].set_xticklabels([f'{tick:.00f}' for tick in band_axes[1].get_xticks()],rotation=45)

band_axes[1].set_xlim(80,300)

    

    #array_ax.plot(np.degrees(tm.array.edge_x).T,np.degrees(tm.array.edge_y).T,
    #              color='k',lw=5e-1)

data_axes[0].legend(handles=handles[::-1])
spec_axes[0].legend(handles=handles[::-1])
band_axes[0].legend(handles=handles[::-1])

    

#fig,ax = plt.subplots(1,1,figsize=(8,8),constrained_layout=True)

#axes[1].scatter(tm.pointing.time,np.degrees(np.abs(tm.atmosphere.aam)))


beam_plot_height = int(np.sqrt(len(tm.atmosphere.depths)))
beam_plot_length = int(np.ceil(len(tm.atmosphere.depths)/beam_plot_height))

#fig,ax = plt.subplots(1,1,figsize=(8,8),constrained_layout=True)#,sharex=True,sharey=True)

fig,axes = plt.subplots(beam_plot_height,beam_plot_length,
                        figsize=(2*beam_plot_length,2*beam_plot_height),constrained_layout=True)

for ilay,depth in enumerate(tm.atmosphere.depths):
    
    ax = axes.ravel()[ilay]

    filt_lim_r = depth*(np.abs(tm.beam_filter_sides[ilay]).max())
    extent = [-filt_lim_r,filt_lim_r,-filt_lim_r,filt_lim_r]
    ax.imshow(tm.beam_filters[ilay],extent=extent)
    ax.grid(b=False)
    #ax.set_xlabel(r'$\theta_x$ (arcmin.)'); ax.set_ylabel(r'$\theta_y$ (arcmin.)')




if tm.array.n < 20:
    
    fig,axes = plt.subplots(beam_plot_height,beam_plot_length,
                            figsize=(2*beam_plot_length,2*beam_plot_height),constrained_layout=True,sharex=True,sharey=True)
    
    fig.suptitle(f'D = {tm.beams.aperture:.02f}m')
    for ilay,depth in enumerate(tm.atmosphere.depths):
        
        iy = ilay % beam_plot_length
        ix = int(ilay / beam_plot_length)
        
        axes[ix,iy].set_title(f'z = {depth:.02f}m')
        axes[ix,iy].plot(np.degrees(tm.array.x+tm.beams_waists[ilay]/depth*np.cos(pt)[:,None]),
                         np.degrees(tm.array.y+tm.beams_waists[ilay]/depth*np.sin(pt)[:,None]),lw=.5)
    




