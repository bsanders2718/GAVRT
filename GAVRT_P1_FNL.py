# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:53:13 2024

@author: bruce
"""
#------------------------Complete Python Program--------------------------------------------------

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy, argparse, os, pdb, csv, yaml
import shutil
import sys,glob

from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy import units as u

import sunpy.map
from sunpy.coordinates import frames, sun

#Get Command Line arguments
#args = sys.argv[1:]

# Set Directories for scan_input, scan_output, png_images, and YAML file. This did not work
#scan_input_dir=args[0]  # Directory that has the raster scan data
#scan_output_dir=args[1] # Directory that the raster scan data is moved to after processed
#png_image_dir=args[2] #Directory that the resulting png images are deposited.
#png_image_aux_dir=args[3] #directory that has the YAML file


#Directories used
scan_input_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\scan_input_dir'  # Directory that has the raster scan data
scan_output_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\scan_output_dir' # Directory that the raster scan data is moved to after processed
png_image_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir' #Directory that the resulting png images are deposited.
png_image_aux_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_aux_dir' #directory that has the YAML file

# DEFINE CONSTANTS
# speed of light [m/s]
C_CONST = 3.e8
# DSS-28 location
DSS28_LOCATION = EarthLocation(lat=35.2383 * u.deg, lon=-116.779 * u.deg, height=1065.382 * u.m)

def strpquotes(csvfile = str):
    """
    Temporary function -- remove all instances of quotes from rastertable .csv file.
    Returns the path+name to new .csv file with quotes removed.
    """
    input_file      = open(csvfile, 'r')
    csvbase, csvext = os.path.splitext(csvfile)
    newcsvfile      = f'{csvbase}_noquotes{csvext}'
    
    # delete file if already exists
    if os.path.exists(newcsvfile):
        os.remove(newcsvfile)
    
    output_file = open(newcsvfile, 'w')
    
    data   = csv.reader(input_file)
    writer = csv.writer(output_file)
    
    for line in data:
        line = str(line)
        #new_line = str.replace(line,'\'[]', '')
        mapping_table = str.maketrans({'\'': '', '[': '', ']': ''})
        new_line = line.translate(mapping_table)
        writer.writerow(new_line.split(','))
    
    input_file.close()
    output_file.close()
    return newcsvfile


def gaussian2D(xy, height, center_x, center_y, width_x, width_y, a0):
    """
    Returns a Gaussian function with the given parameters.
    """
    x, y = xy
    output = height*np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2) + a0
    return output.ravel()


def quietSun(freq_GHz = float):
    """
    Return quiet Sun brightness temperature at specified frequency, freq_GHz.
    Estimates for now (from Stephen White slides). TODO - replace with Zhang et al. 2022 model
    """
    # quiet Sun spectral index
    spIndex = -0.81
    # reference frequency (GHz)
    refFreq = 1.0
    # quiet Sun brightness temperature at reference frequency (K)
    Tb_refFreq = 100000
    return Tb_refFreq * 10**(spIndex * np.log10(freq_GHz/refFreq))
   
    
def beamWidth(freq_GHz = float):
    """
    Return DSS-28 beamwidth in arcmin at specified frequency, freq_GHz
    """
    dishDiameter = 34 # meters
    freq_Hz = freq_GHz * 1e9
    return 1.22 * (C_CONST/freq_Hz) / dishDiameter * (180./np.pi*60.)


def clipCounts(tsrc_3vals):
    """
    Clip middle value if it is more than 1.5 times adjacent bins or less than 0.5 times adjacent bins.
    Replace clipped value with the average of adjacent bins.
    """
    if ((tsrc_3vals[1] > 1.5*tsrc_3vals[0]) and (tsrc_3vals[1] > 1.5*tsrc_3vals[2])) or \
       ((tsrc_3vals[1] < 0.5*tsrc_3vals[0]) and (tsrc_3vals[1] < 0.5*tsrc_3vals[2])):
        return np.mean([tsrc_3vals[0], tsrc_3vals[2]])
    else:
        return tsrc_3vals[1]


def mapInterpolation(xdecoff, decoff, tsrc, convolveGaussian = False, freq_GHz = 3.0):
    """
    Interpolate raster map, calibrate to quiet Sun, and correct pointing offset.
    Image dimensions set to 101 x 101 pixels.
    If convolveGaussian = True, smooth map with beam PSF. Defaults to PSF for 3.0 GHz unless
    specified with {freq_GHz}.
    """
    xdecmin   = np.round(np.min(xdecoff), decimals=1)
    xdecmax   = np.round(np.max(xdecoff), decimals=1)
    decmin    = np.round(np.min(decoff), decimals=1)
    decmax    = np.round(np.max(decoff), decimals=1)
    npixels   = 101
    xi        = np.linspace(xdecmin, xdecmax, npixels)
    deltaxdec = xi[1] - xi[0]
    yi        = np.linspace(decmin, decmax, npixels)
    deltadec  = yi[1] - yi[0]
    xi,yi     = np.meshgrid(xi, yi)
    tsrcmap   = scipy.interpolate.griddata((xdecoff,decoff), tsrc, (xi,yi))
    tsrcmap[np.isnan(tsrcmap)] = 0
    if convolveGaussian:
        # get the PSF size in units of pixels
        xpixelScale = (xdecmax - xdecmin)/npixels * 60  # arcminutes
        ypixelScale = (decmax - decmin)/npixels * 60    # arcminutes
        psfxpix = np.round(beamWidth(freq_GHz) / xpixelScale)
        psfypix = np.round(beamWidth(freq_GHz) / ypixelScale)
        tsrcmap = scipy.ndimage.gaussian_filter(tsrcmap, [psfxpix, psfypix])
    # fit 2D Gaussian to get approximate bounds of quiet Sun (this is to correct for offset in Sun position)
    try:
        params, covar = scipy.optimize.curve_fit(gaussian2D, (xi,yi), tsrcmap.ravel())
    except:
        # curve_fit failed; should incorporate something into the log here about this mapID and channum not producing a valid map
        return False, False, False, False, False
    # within quiet Sun bounds, remove outliers (to remove AR flux)
    maskpos = (xi - params[1])**2. + (yi - params[2])**2. < (params[3] - params[1])**2 + (params[4] - params[2])**2
    maskmean = np.mean(tsrcmap[maskpos])
    maskstd = np.std(tsrcmap[maskpos])
    maskflux = tsrcmap < maskmean+maskstd
    # set ARs equal to quiet Sun, and refit for quiet Sun bounds
    tsrcmap_qs = np.copy(tsrcmap)
    qs = np.median(tsrcmap[maskpos & maskflux])
    tsrcmap_qs[np.where(tsrcmap > maskmean+maskstd)] = qs
    try:
        params2, covar2 = scipy.optimize.curve_fit(gaussian2D, (xi,yi), tsrcmap_qs.ravel())
    except:
        return False, False, False, False, False
    maskpos2 = (xi - params2[1])**2. + (yi - params2[2])**2. < (params2[3] - params2[1])**2 + (params2[4] - params2[2])**2
    # calibrate using quiet Sun
    gain = quietSun(freq_GHz)/qs
    return tsrcmap*gain, xdecoff-params2[1], decoff-params2[2], deltaxdec, deltadec



def getSuncoords(xdecoff, decoff, deltaxdec, deltadec, haSun, decSun, obstimes, obstimes_starttimes, tsrcmap, mapID, chan, freq_GHz = 3.0, outputdir=png_image_dir, contours=False): #Changed outputdir 07/22/2024
    """
    Convert xdecoff,decoff coordinates to heliographic coordinate system, and generate .png and FITS images.
    """
    center_ind     = np.argmin(np.sqrt(xdecoff**2 + decoff**2))
    center_xdec    = xdecoff[center_ind]
    center_dec     = decoff[center_ind]
    sunoffset_xdec = xdecoff[int(len(xdecoff)/2)-1] - center_xdec 
    sunoffset_dec  = decoff[int(len(decoff)/2)-1] - center_dec
    obstime        = obstimes[center_ind]
    starttime      = obstimes_starttimes[center_ind]
    sun_loc        = get_sun(obstime)
    DSS28_gcrs      = SkyCoord(DSS28_LOCATION.get_gcrs(obstimes[int(len(obstimes)/2)-1]))
    reference_coord = SkyCoord(sun_loc.ra - (sunoffset_xdec/np.cos((sunoffset_dec*u.deg).to(u.rad)))*u.deg, 
                               sun_loc.dec + (sunoffset_dec*u.deg), 
                               frame='gcrs', obstime=obstimes[int(len(obstimes)/2)-1], 
                               obsgeoloc=DSS28_gcrs.cartesian, 
                               obsgeovel=DSS28_gcrs.velocity.to_cartesian(),
                               distance=DSS28_gcrs.hcrs.distance)
    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=DSS28_gcrs))
    cdelt2 = ((deltaxdec/np.cos((deltadec*u.deg).to(u.rad)))*u.deg).to(u.arcsec)
    cdelt1 = (deltadec*u.deg).to(u.arcsec)
    P1     = sun.P(obstime)
    new_header = sunpy.map.make_fitswcs_header(tsrcmap, reference_coord_arcsec,
                                               reference_pixel=u.Quantity([int(tsrcmap.shape[0]/2)-1,
                                                                           int(tsrcmap.shape[1]/2)-1]*u.pixel),
                                               scale=u.Quantity([cdelt1,cdelt2]*u.arcsec/u.pix),
                                               rotation_angle=-P1,
                                               wavelength=freq_GHz*u.GHz,
                                               observatory='DSS-28')
    gavrt_map = sunpy.map.Map(tsrcmap, new_header)
    #fig = plt.figure()
    #ax = fig.add_subplot(projection=gavrt_map)
    #gavrt_map.plot(axes=ax, cmap='viridis')
    #gavrt_map.draw_limb(axes=ax)
    #plt.show()
    
    gavrt_map_rotate = gavrt_map.rotate()
    bl = SkyCoord(-1500*u.arcsec, -1500*u.arcsec, frame=gavrt_map_rotate.coordinate_frame)
    tr = SkyCoord(1500*u.arcsec, 1500*u.arcsec, frame=gavrt_map_rotate.coordinate_frame)
    gavrt_submap = gavrt_map_rotate.submap(bl, top_right=tr)
    fig = plt.figure()
    ax = fig.add_subplot(projection=gavrt_submap)
    gavrt_submap.plot(axes=ax, cmap='afmhot')
    gavrt_submap.draw_limb(axes=ax)
    gavrt_submap.draw_grid(axes=ax)
    if contours:
        #gavrt_submap.draw_contours(np.arange(60, 100, 5)*u.percent, axes=ax)
        gavrt_submap.draw_contours(np.logspace(1.5, 1.99, 10)*u.percent, axes=ax)
    cbar = plt.colorbar()
    cbar.set_label(r'$T_b$ (K)', rotation=90)
    
    #fits.writeto(f'{outputdir}/fits/{obstime}_{freq_GHz}.fits', gavrt_map, new_header)
    outimfilename = f'{mapID}_chan{chan:02d}_{str(starttime.datetime).replace(":","-").replace(" ","T")}_{round(freq_GHz*1000)}MHz'
    #gavrt_submap.save(f'{outputdir}/fits/{outimfilename}.fits', overwrite=True)
    plt.savefig(f'{outputdir}/{outimfilename}.png') #changed 07/22/2024
    #plt.show()


def main(rasterfile,logfile,freqfile,outputdir,contours): #Changed on 07/22/2024
    """
    Generate solar maps from raster table.
    Args: <path-to-raster-table/raster-table.txt>
    Derived from Velu's IDL_rastermap.pro and Tushar's Python script.
    
    Example call from Marin's local computer: ipython rastermap.py ../data/rasterTables/raster_92_2024.csv -- --freqfile ../data/logs/scan2024-04-01_doy092.yaml --outputdir ../data/maps/doy092 --contours
    """
    #parser = argparse.ArgumentParser(description="Generate solar maps from raster table.")
    #parser.add_argument("rasterfile", help="Raster table filename.")
    #parser.add_argument("--logfile", type=str, default=None, help="Output log file.")
    #parser.add_argument("--freqfile", type=str, default=None, help="Use frequency-channel mapping in log, rather than using frequency value in rastertable. This is relevant only for data prior to 2024 DOY 110")
    #parser.add_argument("--outputdir", type=str, default=None, help="Specify location to output FITS and .png files. If not specified, will default to creating <fits> and <pngs> directories in current working directory.")
    #parser.add_argument("--contours", action="store_true", default=False, help="If option is set, contours are included on the maps. Default is no contours.")
    #args   = parser.parse_args()
    
    #if not args.logfile:
    #    logfile = f'{os.path.splitext(os.path.abspath(args.rasterfile))[0]}.log'
    #else:
    #    logfile = args.logfile
    
    if not os.path.exists(scan_output_dir):
        os.mkdir(scan_output_dir)
        os.mkdir(f'{scan_output_dir}/fits')
        #os.mkdir(f'{outputdir}') #changed from f'{outputdir}/pngs' on 07/22/2024

    rasterfile = strpquotes(rasterfile)

    # read in raster table
    rastertable = np.genfromtxt(rasterfile, comments=';', names=True, dtype=None, encoding='UTF-8', delimiter=', ')
    # rastertable.dtype
    # rastertable.dtype.names
    # raster_cfg_id  rss_cfg_id  year  doy  utc  epoch  source_id  chan  freq  rate  step  raster_id  raster_cfg_id  year  doy  utc  epoch  xdecoff  decoff  ha  dec  tsrc 
    # raster_cfg_id is equivalent to the map ID;  raster_id is record number equivalent scan sampling number

    # get datetime info
    doy_str      = f"{int(rastertable['year'][0])}/{int(rastertable['doy'][0])}"
    date_str     = datetime.strptime(doy_str, '%Y/%j')
    datestr_arr  = np.array([f"{year}:{doy}:{time}" for year,doy,time in zip(rastertable['year_1'], rastertable['doy_1'], rastertable['utc_1'])])
    datetime_arr = Time(datestr_arr, format='yday')
    
    # get datetime info for raster scan start time
    datestr_arr_starttime  = np.array([f"{year}:{doy}:{time}" for year,doy,time in zip(rastertable['year'], rastertable['doy'], rastertable['utc'])])
    datetime_arr_starttime = Time(datestr_arr_starttime, format='yday')

    # find number of maps and channels observed on this day
    maps_all  = rastertable['raster_cfg_id']
    maps      = np.unique(maps_all)
    chans_all = rastertable['chan_1']
    chans     = np.unique(chans_all)
    # get mapping between frequency and channel number
    inds_arr = []
    if not freqfile:
        freqs_GHz = rastertable['freq']/1000.   # in GHz
        for mapID in maps:
            inds = np.where(rastertable['raster_cfg_id'] == mapID)
            inds_arr.append(inds)
    else:
        with open(freqfile) as f:
            freqmapping = yaml.safe_load(f)
        freqs_GHz = np.array([freqmapping[str(channum)]/1000. for channum in chans_all])
        for mapID in maps:
            for chan in chans:
                inds = np.where((rastertable['raster_cfg_id'] == mapID) & (chans_all == chan))
                inds_arr.append(inds)

    # Loop over all maps by map_ID and channel number.
    for inds in inds_arr:
        freq_GHz    = np.unique(freqs_GHz[inds])[0]
        mapID       = np.unique(maps_all[inds])[0]
        chan        = np.unique(chans_all[inds])[0]
        utc_arr     = datetime_arr[inds]
        utc_arr_starttime = datetime_arr_starttime[inds]
        tsrc_arr    = rastertable['top'][inds]
        tsrc_arr_clipped = scipy.ndimage.generic_filter(tsrc_arr, clipCounts, 3)
        xdecoff_arr = rastertable['xdecoff'][inds]
        decoff_arr  = rastertable['decoff'][inds]
        ha_arr      = rastertable['ha'][inds]   # decimal degrees
        dec_arr     = rastertable['dec'][inds]  # decimal degrees
        tsrcmap, xdecoff_shift, decoff_shift, deltaxdec, deltadec = mapInterpolation(xdecoff_arr, decoff_arr, tsrc_arr_clipped,
                                                                                     convolveGaussian=False, freq_GHz = freq_GHz)
        if not isinstance(tsrcmap, np.ndarray):
            continue
        getSuncoords(xdecoff_shift, decoff_shift, deltaxdec, deltadec, ha_arr, dec_arr, 
                     utc_arr, utc_arr_starttime, tsrcmap, mapID, chan, freq_GHz, outputdir=png_image_dir,
                     contours=contours)
        #plt.ion()
        #plt.close('all')
        #plt.figure()
        #plt.imshow(tsrcmap, origin='lower', extent=(xdecoff_shift.min(), xdecoff_shift.max(), decoff_shift.min(), decoff_shift.max()))
        #plt.colorbar()
        #pdb.set_trace()

# Loop Through files and output
for file_name in os.listdir(scan_input_dir): #Windows Code
 
    try: 
        main(rasterfile=scan_input_dir+'\\'+file_name,logfile='tmp.txt',freqfile=png_image_aux_dir+'\\scan2024-04-03_doy094.yaml',outputdir=png_image_dir,contours=False)  #Windows Code
        shutil.move(scan_input_dir+'\\'+file_name,scan_output_dir+'\\'+file_name) 
        
        print(file_name)
    except:
        main(rasterfile=scan_input_dir+'\\'+file_name,logfile='tmp.txt',freqfile=False,outputdir=png_image_dir,contours=False) #Windows Code
        shutil.move(scan_input_dir+'\\'+file_name,scan_output_dir+'\\'+file_name)
        
        print(file_name)
        
    #finally delete all "No QUotes" files
    os.chdir(scan_input_dir)
    for f in glob.glob('*_noquotes.csv'):
        os.remove(f)