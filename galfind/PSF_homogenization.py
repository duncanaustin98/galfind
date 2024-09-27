#     kernel_dir,
#     match_band="F444W",
#     override_bands=None,
#     use_fft_conv=True,
#     overwrite=False,
#     update_default_dictionaries=True,
# ):
#     galfind_logger.warning(
#         "Data.convolve_images easily sped up without the use of Instrument.instrument_from_band!"
#     )
#     """Adapted from aperpy - https://github.com/astrowhit/aperpy/"""
#     if override_bands is not None:
#         bands = override_bands
#     else:
#         bands = self.instrument.band_names
#     outdir = f"{config['DEFAULT']['GALFIND_WORK']}/PSF_Matched_Images/{self.version}/{self.instrument.instrument_from_band(bands[0]).name}/{self.survey}"
#     self.im_psf_matched_dir = outdir
#     if use_fft_conv:
#         convolve_func = convolve_fft
#         convolve_kwargs = {"allow_huge": True}
#     else:
#         convolve_func = convolve
#         convolve_kwargs = {}

#     # for filename in sci_paths:
#     im_paths_matched = {}
#     wht_paths_matched = {}
#     rms_err_paths_matched = {}

#     for band in bands:
#         im_filename = self.im_paths[band]
#         wht_filename = self.wht_paths[band]
#         err_filename = self.rms_err_paths[band]
#         same_file = im_filename == wht_filename == err_filename
#         outnames = []

#         if not os.path.exists(outdir):
#             os.makedirs(outdir)

#         if same_file:
#             print(
#                 "WHT, SCI, and ERR are the same file!. Output will be written to the same file."
#             )
#             outname = im_filename.replace(
#                 ".fits", f"_{match_band}-matched.fits"
#             ).replace(os.path.dirname(im_filename), outdir)
#             outnames.append(outname)
#             outsciname = outwhtname = outerrname = outname
#             full_hdu = fits.open(im_filename)
#         else:
#             outsciname = im_filename.replace(
#                 ".fits", f"_sci_{match_band}-matched.fits"
#             ).replace(os.path.dirname(im_filename), outdir)
#             outwhtname = wht_filename.replace(
#                 ".fits", f"_wht_{match_band}-matched.fits"
#             ).replace(os.path.dirname(wht_filename), outdir)
#             outerrname = err_filename.replace(
#                 ".fits", f"_err_{match_band}-matched.fits"
#             ).replace(os.path.dirname(err_filename), outdir)
#             outnames.append(outsciname)
#             outnames.append(outwhtname)
#             outnames.append(outerrname)

#         im_paths_matched[band] = outsciname
#         wht_paths_matched[band] = outwhtname
#         rms_err_paths_matched[band] = outerrname

#         skip = False
#         for outname in outnames:
#             if os.path.exists(outname) and not overwrite:
#                 print(outsciname, outwhtname)
#                 print("Convolved images exist, I will not overwrite")
#                 skip = True

#         if not skip:
#             print("  science image: ", im_filename)
#             print("  weight image: ", wht_filename)
#             print("  error image: ", err_filename)
#             hdul = fits.open(im_filename)
#             hdul_wht = fits.open(wht_filename)

#             if err_filename != "":
#                 hdul_err = fits.open(err_filename)

#             if band != match_band:
#                 print(f"  PSF-matching sci {band} to {match_band}")
#                 tstart = time.time()
#                 fn_kernel = os.path.join(kernel_dir, f"{band}_kernel.fits")
#                 print("  using kernel ", fn_kernel.split("/")[-1])
#                 kernel = fits.getdata(fn_kernel)
#                 kernel /= np.sum(kernel)

#                 if same_file:
#                     wht_ext = "WHT"
#                 else:
#                     wht_ext = 0
#                 weight = hdul_wht[wht_ext].data
#                 if not same_file:
#                     out_hdul = fits.HDUList([])
#                 else:
#                     out_hdul = full_hdu.copy()
#                 if overwrite or not os.path.exists(outsciname):
#                     print("Running science image convolution...")
#                     if same_file:
#                         sci_ext = "SCI"
#                     else:
#                         sci_ext = 0

#                     sci = hdul[sci_ext].data
#                     data = convolve_func(
#                         sci, kernel, **convolve_kwargs
#                     ).astype(np.float32)
#                     data[weight == 0] = 0.0
#                     print("convolved...")

#                     out_hdu = fits.PrimaryHDU(
#                         data, header=hdul[sci_ext].header
#                     )
#                     out_hdu.name = "SCI"
#                     out_hdu.header["HISTORY"] = (
#                         f"Convolved with {match_band} kernel"
#                     )
#                     out_hdu.header["HISTORY2"] = (
#                         f"Convolution kernel path: {fn_kernel}"
#                     )
#                     if same_file:
#                         out_hdul[sci_ext].data = out_hdu.data
#                         out_hdul[sci_ext].header["HISTORY"] = (
#                             f"Convolved with {match_band} kernel"
#                         )
#                         out_hdul[sci_ext].header["HISTORY2"] = (
#                             f"Convolution kernel path: {fn_kernel}"
#                         )
#                     else:
#                         out_hdul.append(out_hdu)
#                         out_hdul.writeto(outsciname, overwrite=True)
#                         print("Wrote file to ", outsciname)
#                         out_hdul = fits.HDUList([])

#                 else:
#                     print(outsciname)
#                     print(
#                         f"{band.upper()} convolved science image exists, I will not overwrite"
#                     )

#                 hdul.close()

#                 if overwrite or not os.path.exists(outwhtname):
#                     print("Running weight image convolution...")
#                     err = np.where(weight == 0, 0, 1 / np.sqrt(weight))
#                     err_conv = convolve_func(
#                         err, kernel, **convolve_kwargs
#                     ).astype(np.float32)
#                     data = np.where(err_conv == 0, 0, 1.0 / (err_conv**2))
#                     data[weight == 0] = 0.0

#                     out_hdu_wht = fits.PrimaryHDU(
#                         data, header=hdul_wht[wht_ext].header
#                     )
#                     out_hdu_wht.name = "WHT"
#                     out_hdu_wht.header["HISTORY"] = (
#                         f"Convolved with {match_band} kernel"
#                     )

#                     if same_file:
#                         out_hdul[wht_ext].data = out_hdu_wht.data
#                         out_hdul[wht_ext].header["HISTORY"] = (
#                             f"Convolved with {match_band} kernel"
#                         )
#                     else:
#                         out_hdul.append(out_hdu_wht)
#                         out_hdul.writeto(outwhtname, overwrite=True)
#                         print("Wrote file to ", outwhtname)
#                         out_hdul = fits.HDUList([])

#                 else:
#                     print(outwhtname)
#                     print(
#                         f"{band.upper()} convolved weight image exists, I will not overwrite"
#                     )

#                 hdul_wht.close()

#                 if outerrname != "" and (
#                     overwrite or not os.path.exists(outerrname)
#                 ):
#                     print("Running error image convolution...")

#                     data = convolve_func(
#                         hdul_err["ERR"].data, kernel, **convolve_kwargs
#                     ).astype(np.float32)
#                     data[weight == 0] = 0.0

#                     out_hdu_err = fits.PrimaryHDU(
#                         data, header=hdul_err["ERR"].header
#                     )
#                     out_hdu_err.name = "ERR"
#                     out_hdu_err.header["HISTORY"] = (
#                         f"Convolved with {match_band} kernel"
#                     )
#                     if same_file:
#                         out_hdul["ERR"].data = out_hdu_err.data
#                         out_hdul["ERR"].header["HISTORY"] = (
#                             f"Convolved with {match_band} kernel"
#                         )
#                     else:
#                         out_hdul.append(out_hdu_err)
#                         out_hdul.writeto(outerrname, overwrite=True)
#                         print("Wrote file to ", outerrname)
#                         out_hdul = fits.HDUList([])

#                     hdul_err.close()

#                 print(f"Finished in {time.time()-tstart:2.2f}s")

#                 if same_file and len(out_hdul) > 1:
#                     out_hdul.writeto(outname, overwrite=True)
#                 else:
#                     print("Not writing empty HDU")
#             else:
#                 outsciname = im_filename.replace(
#                     ".fits", f"_sci_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(im_filename), outdir)
#                 outwhtname = wht_filename.replace(
#                     ".fits", f"_wht_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(wht_filename), outdir)
#                 outerrname = err_filename.replace(
#                     ".fits", f"_err_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(err_filename), outdir)
#                 outname = im_filename.replace(
#                     ".fits", f"_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(im_filename), outdir)

#                 if same_file:
#                     hdul.writeto(outname, overwrite=True)
#                     print(hdul.info())
#                 else:
#                     hdul.writeto(outsciname, overwrite=True)
#                     hdul_wht.writeto(outwhtname, overwrite=True)
#                     hdul_wht.close()
#                     if err_filename != "":
#                         hdul_err.writeto(outerrname, overwrite=True)
#                         hdul_err.close()
#                 print("Written files to ", outname)
#                 hdul.close()

#         # Update paths in self
#         if update_default_dictionaries:
#             if same_file:
#                 self.im_paths[band] = outname
#                 self.wht_paths[band] = outname
#                 self.rms_err_paths[band] = outname
#             else:
#                 self.im_paths[band] = outsciname
#                 self.wht_paths[band] = outwhtname
#                 self.rms_err_paths[band] = outerrname

#     return im_paths_matched, wht_paths_matched, rms_err_paths_matched
