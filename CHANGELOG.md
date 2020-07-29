ErowCloudViewer Version History
============================

v2.10.3 (Zephyrus) - (in development)
----------------------

- Enhancements
  - Speed up the Fit Sphere tool and point picking in the Registration Align (point pairs picking) tool
  - Translation:
	- new (argentinian) Spanish translation

- Bug fixes
  - Command line:
    - the 'EXTRACT_VERTICES' option was actually deleting the extracted vertices right after extracting them, causing a crash when trying to access them later :| (#847)
    - fix handling of SF indices in SF_ARITHMETIC and COMMAND_SF_OP
    - the COMMAND_ICP_ROT option of the ICP command line tool was ignored (#884)
    - when loading a BIN file from the command line, only the first-level clouds were considered
  - Fix loading LAS files with paths containing multi-byte characters when using PDAL (#869)
  - When saving a cloud read from LAS 1.0 let PDAL choose the default LAS version (#874)
  - Fix potential crash or use of incorrect data when comparing clouds (#871)
  - Fix potential crash when quitting or switching displays
  - Quitting the "Section extraction tool" (and probably any tool that uses a temporary 3D view, such as the Align tool) would break the picking hub mechanism (preventing the user from picking points typically) (#886)
  - Fix the camera name being displayed in the wrong place (#902)
  - The layers management of the Rasterize tool was partially broken
  - the C2C/C2M distance computation tool called through the command line was always displaying progress dialogs even in SILENT mode
  - the ICP registration tool called through the command line was always displaying progress dialogs even in SILENT mode
  - Fix potential crash with qCSF (see github issue #909)
  - In some cases, the (subsampled) core points cloud was not exported and saved at the end of the call to M3C2 through the command line
  - Some points were incorrectly removed by the 'Clean > Noise filer' method (parallelism issue)
  - The radius was not updated during the refinement pass of the Sphere fitting algorithm  (i.e. the final radius was not optimal)

v2.10.2 (Zephyrus) - 24/02/2019
----------------------

- Bug fixes
  - Rasterize tool:
    - interpolating empty cells with the 'resample input cloud' option enabled would make CC crash
    - change layout so it works better on lower-resolution monitors
  - Command line:
    - the 'EXTRACT_VERTICES' option was not accessible
    - calling the -RASTERIZE option would cause an infinite loop
    - the Global Shift & Scale information of the input cloud was not transferred to the output cloud of the -RASTERIZE tool
  - glitch fix: the 3D window was not properly updated after rendering the screen as a file with a zoom > 1
  - glitch fix: the name of the entity was not displayed at the right place when rendering the screen as a file with a zoom > 1
  - the Surface and Volume Density features were potentially outputting incorrect values (the wrong source scalar field was used when applying the dimensional scale!)
  - the chosen octree level could be sub-optimal in some very particular cases
  - E57 pinhole images:
    - fix sensor array information (it was displaying total image size for the width of the image)
    - fix pixel width & height


- Translations
  - updated Russian translation (thanks to Eugene Kalabin)
  - added Japanese translation (thanks to the translators at CCCP)


- macOS Note
  - I (Andy) had to update ffmpeg, which is used by the animation plugin, for this patch release. Normally I would wait for 2.11, but homebrew changed their policies and started including everything in their build, so I can no longer use it. The good news is that compiling ffmpeg myself and statically linking shaves about 30 MB off the size of ErowCloudViewer.app...
  - it has been reported that this fixes a potential crash in ffmpeg's libavutil.56.dylib

v2.10.1 (Zephyrus) - 01/16/2019
----------------------

- Bug fixes:

  - writing E57 files was broken
  - an exception was being thrown when you close CC after saving an ASCII file (#834)

v2.10 (Zephyrus) - 01/06/2019
----------------------

- new features:

	* Edit > Polyline > Sample points
		- to regularly samples points on one or several polylines

	* New set of geometrical features to compute on clouds:
		- Tools > Other > Compute geometric features
		- features are all based on locally computed eigen values:
			* sum of eigen values
			* omnivariance
			* eigenentropy
			* anisotropy
			* planarity
			* linearity
			* PCA1
			* PCA2
			* surface variation
			* sphericity
			* verticality
		- most of the features are defined in "Contour detection in unstructured 3D point clouds", Hackel et al, 2016

	* Localization support
		- Display > Language translation
		- currently supported languages:
			* English (default)
			* Brazilian portuguese (partial)
			* French (very partial)
			* Russian (partial)
		- volunteers are welcome: https://www.CLOUDVIEWER .org/forum/viewtopic.php?t=1444

- enhancements:

	* Roughness, Density and Curvature can now all be computed via the new 'Tools > Other > Compute geometric features' menu
		(Approx density can't be computed anymore)

	* Global Shift & Scale dialog
		- new option "Preserve global shift on save" to directly control whether the global coordinates should be preserved
			at export time or simply forgotten

	* The 'Display > Lock vertical rotation' option has been renamed 'Display > Lock rotation about an axis' (Shortcut: L)
		- CC will now ask for the rotation axis to be locked (default: Z)

	* The M3C2 plugin can now be called from the command line:
		- the first time you'll need the configuration file saved with the GUI tool
			(Use the 'Save parameters to file' button in the bottom-left corner of the M3C2 dialog --> the floppy icon)
		- then load 2 files (cloud 1 and cloud2)
		- optionally load a 3rd cloud that will be used as core points
		- and eventually call the -M3C2 option with the parameter file as argument:
			ErowCloudViewer -O cloud1 -O cloud2 (-O core_points) -M3C2 parameters_file
		- new option to use the core points cloud normals (if any)

	* The Canupo plugin is now open-source!
		- Thanks (once again) to Dimitri Lague for this great contribution
		- the code is here: https://github.com/ErowCloudViewer/ErowCloudViewer/tree/master/plugins/core/qCanupo

	* The "Classify" option of the Canupo plugin can now be called from the command line:
		- you'll need a trained classifier (.prm file)
		- main option: -CANUPO_CLASSIFY classifier.prm
		- confidence threshold:
			* -USE_CONFIDENCE {threshold}  (threshold must be between 0 and 1)
			* (use the 'SET_ACTIVE_SF' after loading a cloud to set the active scalar field if
				you want it to be used to refine the classification)
		- syntax:
			ErowCloudViewer -O cloud1 ... -O cloudN -CANUPO_CLASSIFY (-USE_CONFIDENCE 0.9) classifier.prm

	* Labels can now be imported from ASCII files:
		- new column role in the ASCII loading dialog: "Labels"
		- labels can be created from textual or numerical columns
		- one "2D label" entity is created per point (don't try to load too many of them ;)
		- labels are displayed in 3D by default (i.e. next to each point), but they can also be displayed in 2D (see the dedicated check-box)

	* FBX units:
		- default FBX units are 'cm'
		- if a FBX file with other units is imported, CC will now store this information as meta-data and will set it correctly
			if the corresponding meshes are exported as FBX again

	* Command line mode:
		- scalar field convert to RGB:
			* '-SF_CONVERT_TO_RGB {mixWithExistingColors bool}'
		- scalar field set color scale:
			* '-SF_COLOR_SCALE {filename}'
		- extract all loaded mesh vertices as standalone 'clouds' (the mesh is discarded)
			* '-EXTRACT_VERTICES'
		- remove all scan grids
			* '-REMOVE_SCAN_GRIDS'
		- new sub-option of 'SAVE_CLOUDS' to set the output filename(s) (e.g. -SAVE_CLOUDS FILE "cloud1.bin cloud2.bin ..."
		- new options for the 'OCTREE_NORMALS' (thanks to Michael Barnes):
			* '-ORIENT' to specify a default orientation hint:
				- PLUS_ZERO
				- MINUS_ZERO
				- PLUS_BARYCENTER
				- MINUS_BARYCENTER
				- PLUS_X
				- MINUS_X
				- PLUS_Y
				- MINUS_Y
				- PLUS_Z
				- MINUS_Z
				- PREVIOUS
			* '-MODEL' to specify the local model:
				- LS
				- TRI
				- QUADRIC

	* Unroll tool:
		- the cylindrical unrolling can be performed inside an arbitrary angular range (between -3600 and +3600 degrees)
		- this means that the shape can be unrolled on more than 360 degrees, and from an arbitrary starting orientation

	* New options (Display > Display options):
		- the user can now control whether normals should be enabled on loaded clouds by default or not (default state is now 'off')
		- the user can now control whether load and save dialogs should be native ones or generic Qt dialogs

	* Normals:
		- ergonomics of 'Normals > compute' dialog have been (hopefully) enhanced
		- normals can now be oriented toward a sensor even if there's no grid associated to the point cloud.
		- the Normal Orientation algorithm based on the Minimum Spanning Tree now uses much less memory (~1/10)

	* PCV:
		- the PCV plugin can now be applied on several clouds (batch mode)

	* LAS I/O:
		- ErowCloudViewer can now read and save extra dimensions (for any file version) - see https://github.com/ErowCloudViewer/ErowCloudViewer/pull/666

	* E57:
		- the E57 plugin now uses [libE57Format] (https://github.com/asmaloney/libE57Format) which is a fork of the old E57RefImpl
		- if you compile ErowCloudViewer with the E57 plugin, you will need to use this new lib and change some CMake options to point at it - specifically **OPTION_USE_LIBE57FORMAT** and **LIBE57FORMAT_INSTALL_DIR**
		- the E57 plugin is now available on macOS

	* RDS (Riegl)
		- the reflectance scalar field read from RDS file should now have correct values (in dB)

	* SHP:
		- improved support thanks to T. Montaigu (saving and loading Multipatch entities, code refactoring, unit tests, etc.)

	* Cross section tool:
		- can now be started with a group of entities (no need to select the entities inside anymore)
		- produces less warnings

	* Plugins (General):
		- the "About Plugins" dialog was rewritten to provide more information about installed plugins and to include I/O and GL plugins.
		- [macOS] the "About Plugins..." menu item was moved from the Help menu to the Application menu.
		- added several fields to the plugin interface: authors, maintainers, and reference links.
		- I/O plugins now have the option to return a list of filters using a new method *getFilters()* (so one plugin can handle multiple file extensions)
		- moved support for several less frequently used file formats to a new plugin called qAdditionalIO
			- Snavely's Bundler output (*.out)
			- Clouds + calibrated images [meta][ascii] (*.icm)
			- Point + Normal cloud (*.pn)
			- Clouds + sensor info. [meta][ascii] (*.pov)
			- Point + Value cloud (*.pv)
			- Salome Hydro polylines (*.poly)
			- SinusX curve (*.sx)
			- Mensi Soisic cloud (*.soi)

	* Misc:
		- some loading dialogs 'Apply all' button will only apply to the set of selected files (ASCII, PLY and LAS)
		- the trace polyline tool will now use the Global Shift & Scale information of the first clicked entity
		- when calling the 'Edit > Edit Shift & Scale' dialog, the precision of the fields of the shift vector is now 6 digits
			(so as to let the user manually "geo-reference" a cloud)
		- the ASCII loading dialog can now load up to 512 columns (i.e. almost as many scalar fields ;). And it shouldn't become huge if
			there are too many columns or characters in the header line!

- bug fixes:

	* subsampling with a radius dependent on the active scalar field could make CC stall when dealing with negative values
	* point picking was performed on each click, even when double-clicking. This could actually prevent the double-click from
		being recognized as such (as the picking could be too slow!)
	* command line mode: when loading at least two LAS files with the 'GLOBAL_SHIFT AUTO' option, if the LAS files had different AND small LAS Shift
	* point picking on a mesh (i.e. mainly in the point-pair based registration tool) could select the wrong point on the triangle, or even a wrong triangle
	* raster I/O: when importing a raster file, the corresponding point cloud was shifted of half a pixel
	* the RASTERIZE command line could make CC crash at the end of the process
	* hitting the 'Apply all' button of the ASCII open dialog would not restore the previous load configuration correctly in all cases
		(the header line may not be extracted the second time, etc.)
	* align tool: large coordinates of manually input points were rounded off (only when displayed)
	* when applying an orthographic viewport while the 'stereo' mode is enabled, the stereo mode was broken (now a warning message is displayed and
		the stereo mode is automatically disabled)
	* the global shift along vertical dimension (e.g. Z) was not applied when exporting a raster grid to a raster file (geotiff)
	* the 2.5D Volume calculation tool was ignoring the strategy for filling the empty cells of the 'ceil' cloud (it was always using the 'ground' setting)
	* [macOS] fixed the squished text in the Matrix and Axis/Angle sections of the transformation history section of the properties
	* [macOS] fixed squished menus in the properties editor
	* the application options (i.e. only whether the normals should be displayed or not at loading time) were not saved!
	* DXF files generated by the qSRA plugin were broken (same bug as the DXF filter in version 2.9)
	* the OCTREE_NORMALS command was saving a file whatever the state of the AUTO_SAVE option
	* the Align tools could make CC crash when applying the alignment matrix (if the octree below the aligned entity was visible in the DB tree)
	* the clouds and contour lines generated by the Rasterize tool were shifted of half a cell
	* in some cases, merging a mesh with materials with a mesh without could make CC crash
	* command line mode: the VOLUME command parser would loop indefinitely if other commands were appended after its own options + it was ignoring the AUTO_SAVE state.
	* some files saved with version 2.6 to 2.9 and containing quadric primitives or projective camera sensors could not be loaded properly since the version 2.10.alpha of May 2018
	* for a mysterious reason, the FWF_SAVE_CLOUDS command was not accessible anymore...
	* when computing C2C distances, and using both a 2.5D Triangulation local model and the 'split distances along X, Y and Z' option, the split distances could be wrong in some times

v2.9.1 (Omnia) - 11/03/2019
----------------------

- enhancements:

	* Primitive factory
		- sphere center can now be set before its creation (either manually, or via the clipboard if the string is 'x y z')

- Bug fixes:

	* DXF export was broken (styles table was not properly declared)
	* PLY files with texture indexes were not correctly read

v2.9 (Omnia) - 10/22/2019
----------------------

- New features:

	* New plugin: qCompass
		- structural geology toolbox for the interpretation and analysis of virtual outcrop models (by Sam Thiele)
		- see http://www.CLOUDVIEWER .org/doc/wiki/index.php?title=Compass_(plugin)

	* 3D view pivot management:
		- new option to position the pivot point automatically on the point currently at the screen center (dynamic update)
			(now the default behavior, can be toggled thanks to the dedicated icon in the 'Viewing tools' toolbar or the 'Shift + P' shortcut)
		- double clicking on the 3D view will also reposition the pivot point on the point under the cursor
		- the state of this option is automatically saved and restored when CC starts

	* New tool to import scalar fields from one cloud to another: 'Edit > SFs > Interpolate from another entity'
		- 3 neighbor extraction methods are supported (nearest neighbor, inside a sphere or with a given number of neighbors)
		- 3 algorithms are available: average, median and weighted average

	* New sub-menu 'Tools > Batch export'
		- 'Export cloud info' (formerly in the 'Sand-box' sub-menu)
			* exports various pieces of information about selected clouds in a CSV file
			* Name, point count, barycenter
			+ for each scalar field: name, mean value, std. dev. and sum
		- 'Export plane info'
			* exports various pieces of information about selected planes in a CSV file
			* Name, width, height, center, normal, dip and dip direction

	* New interactor to change the default line width (via the 'hot zone' in the upper-left corner of 3D views)

	* New option: 'Display > Show cursor coordinates'
		- if activated, the position of the mouse cursor relatively to the 3D view is constantly displayed
		- the 2D position (in pixels) is always displayed
		- the 3D position of the point below the cursor is displayed if possible

	* New shortcut: P (pick rotation center)

- enhancements:

	* When a picking operation is active, the ESC key will cancel it.

	* qBroom plugin:
		- now has a wiki documentation: http://www.CLOUDVIEWER .org/doc/wiki/index.php?title=Virtual_broom_(plugin)

	* qAnimation plugin:
		- new output option 'zoom' (alternative to the existing 'super resolution' option)
		- the plugin doesn't spam the Console at each frame if the 'super resolution' option is > 1 ;)

	* M3C2 plugin:
		- "Precision Maps" support added (as described in "3D uncertainty-based topographic change detection with SfM
			photogrammetry: precision maps for ground control and directly georeferenced surveys" by James et al.)
		- Allows for the computation of the uncertainty based on precision scalar fields (standard deviation along X, Y and Z)
			instead of the cloud local roughness

	* 'Unroll' tool:
		- new cone 'unroll' mode (the true 'unroll' mode - the other one has been renamed 'Straightened cone' ;)
		- option to export the deviation scalar-field (deviation to the theoretical cylinder / cone)
		- dialog parameters are now saved in persistent settings

	* Plugins can now be called in command line mode
		(the 'ccPluginInterface::registerCommands' method must be reimplemented)
		(someone still needs to do the job for each plugin ;)

	* Trace polyline tool
		- the tool now works on meshes
		- Holding CTRL while pressing the right mouse button will pan the view instead of closing the polyline
		- new 'Continue' button, in case the user has mistakenly closed the polyline and wants to continue

	* Command line mode
		- the Rasterize tool is now accessible via the command line:
			* '-RASTERIZE -GRID_STEP {value}'
			* additional options are:
				-VERT_DIR {0=X/1=Y/2=Z} - default is Z
				-EMPTY_FILL {MIN_H/MAX_H/CUSTOM_H/INTERP} - default is 'leave cells empty'
				-CUSTOM_HEIGHT {value} - to define the custom height filling value if the 'CUSTOM_H' strategy is used (see above)
				-PROJ {MIN/AVG/MAX} - default is AVG (average)
				-SF_PROJ {MIN/AVG/MAX} - default is AVG (average)
				-OUTPUT_CLOUD - to output the result as a cloud (default if no other output format is defined)
				-OUTPUT_MESH - to output the result as a mesh
				-OUTPUT_RASTER_Z - to output the result as a geotiff raster (altitudes + all SFs by default, no RGB)
				-OUTPUT_RASTER_RGB - to output the result as a geotiff raster (RGB)
				-RESAMPLE - to resample the input cloud instead of generating a regular cloud (or mesh)
			* if OUTPUT_CLOUD and/or OUTPUT_MESH options are selected, the resulting entities are kept in memory.
				Moreover if OUTPUT_CLOUD is selected, the resulting raster will replace the original cloud.
		- 2.5D Volume Calculation tool
			* '-VOLUME -GRID_STEP {...} etc.' (see the wiki for more details)
		- Export coord. to SF
			* '-COORD_TO_SF {X, Y or Z}'
		- Compute unstructured cloud normals:
			* '-OCTREE_NORMALS {radius}'
			* for now the local model is 'Height Function' and no default orientation is specified
		- Clear normals
			* '-CLEAR_NORMALS'
		- New mesh merging option
			* '-MERGE_MESHES'
		- Compute mesh volume:
			* '-MESH_VOLUME'
			* optional argument: '-TO_FILE {filename}' to output the volume(s) in a file
		- LAS files:
			* when loading LAS files without any specification about Global Shift, no shift will be applied, not even the LAS file internal 'shift' (to avoid confusion)
			* however, it is highly recommanded to always specifiy a Global Shift (AUTO or a specific vector) to avoid losing precision when dealing with big coordinates!
		- Other improvements:
			* the progress bar shouldn't appear anymore when loading / saving a file with 'SILENT' mode enabled
			* the ASCII loading dialog shouldn't appear anymore in 'SILENT' mode (only if CC really can't guess anything)
			* the default timestamp resolution has been increased (with milliseconds) in order to avoid overwriting files
				when saving very small file (too quickly!)

	* Rasterize tool
		- contour lines generation is now based on GDAL (more robust, proper handling of empty cells, etc.)
		- new option to re-project contour lines computed on a scalar field (i.e. a layer other than the altitudes)
			on the altitudes layer
		- the grid step bounds have been widened (between 1E-5 and 1E+5)

	* Edit > SF > Compute Stat. params
		- the RMS of the active SF is now automatically computed and displayed in the Console

	* PLY I/O filter
		- now supports quads (quads are loaded as 2 triangles)

	* DXF I/O filter
		- now based on dxflib 3.17.0
		- point clouds can now be exported to DXF (the number of points should remain very limited)
		- see fixed bugs below

	* LAS I/O filter
		- the 'Spatial Reference System' of LAS files is now stored as meta-data and restored
			when exporting the cloud as a LAS/LAZ file.

	* [Windows] qLAS_FWF:
		- the plugin (based on LASlib) can now load most of the standard LAS fields
		- the plugin can now save files (with or without waveforms)
		- the plugin can now be called in command line mode:
			-FWF_O: open a LAS 1.3+ file
			-FWF_SAVE_CLOUDS: save cloud(s) to LAS 1.3+ file(s) (options are 'ALL_AT_ONCE' and 'COMPRESSED' to save LAZ files instead of LAS)

	* New method: 'Edit > Waveforms > Compress FWF data'
		- To compress FWF data associated to a cloud (useful after a manual segmentation for instance
			as the FWF data is shared between clouds and remains complete by default)
		- Compression is done automatically when saving a cloud with the 'LAS 1.3 / 1.4' filter (QLAS_FWF_IO_PLUGIN)
			(but it's not done when saving the entity as a BIN file)

	* Oculus support
		- CC now displays in the current 3D view the mirror image of what is displayed in the headset
		- using SDK 1.15

	* Point List Picking tool
		- the list can now be exported as a 'global index, x, y, z' text file

	* Scale / Multiply dialog:
		- new option to use the same scale for all dimensions
		- new option to apply the scale to the 'Global shift' (or not)

	* New Menu Entry: 'Edit > Grid > Delete scan grids'
		- scan grids associated to a cloud can now be deleted (to save space when saving the cloud to a BIN file for instance)

	* qEllipser plugin:
		- option to export the image as a (potentially scaled) point cloud

	* Normal computation tool:
		- new algorithm to compute the normals based on scan grids (faster, and more robust)
		- the 'kernel size' parameter is replaced by 'the minimum angle of triangles' used in the internal triangulation process
		- Plane and Quadric modes will now automatically increase the radius adaptively to reach a minimum number of points and to avoid creating 'zero' (invalid) normals

	* Edit the scalar value of a single point
		- create a label on the point (SHIFT + click)
		- make sure a scalar field is active
		- right click on the label entry in the DB tree and select 'Edit scalar value'

	* Merge (clouds)
		- new option to generate a scalar field with the index of the original cloud for each point

	* Other
		- color scales are now listed in alphabetical order
		- polylines exported from the 'Interactive Segmentation' tool will now use the same Global Shift as the segmented entity(ies)
		- when changing the dip and dip direction of plane parallel with XY, the resulting plane shouldn't rotate in an arbitrary way anymore
		- the filter and single-button plugin toolbars are now on the right side of the window by default (to reset to the default layouts, use "Reset all GUI element positions" at the bottom of the Display menu)
		- the Plane edition dialog now lest the user specify the normal plane in addition to its dip and dip direction
		- new 'Clone' icon with a colored background so as to more clearly spot when the icon is enabled (Nyan sheep!)
		- now using PoissonRecon 9.011
		- the default maximum point size and maximum line width increased to 16 pixels

- Bug fixes:
	* STL files are now output by default in BINARY mode in command line mode (no more annoying dialog)
	* when computing distances, the octree could be modified but the LOD structure was not updated
		(resulting in potentially heavy display artifacts)
	* glitch fix: the 'SF > Gradient' tool was mistakenly renaming the input scalar field ('.gradient' appended)
	* glitch fix: the picking process was ignoring the fact that meshes could be displayed in wireframe mode (they are now ignored in this case)
	* command line 'CROSS_SECTION' option: the repetition of the cuts (<RepeatDim> option) could be incomplete in some cases (some chunks were missing)
	* raster loading: rasters loaded as clouds were shifted of half a pixel
	* the 'Edit > Sensors > Camera > Create' function was broken (input parameters were ignored)
	* merging clouds with FWF data would duplicate the waveforms of the first one
	* invalid lines in ASCII (text) files could be considered as a valid point with coordinates (0, 0, 0)
	* Point-pair based alignment tool:
		- extracting spheres on a cloud with Global Shift would create the sphere in the global coordinate system instead of the local one (i.e. the sphere was not visible)
		- deleting a point would remove all the detected spheres
	* The FARO I/O plugin was associating a wrong transformation to the scan grids, resulting in weird results when computing normals or constructing a mesh based on scan grids
	* When editing only the dip / dip direction of a plane, the rotation was not made about the plane center
	* qSRA plugin: profile polyline automatically generated from cylinders or cone were shifted (half of the cylinder/cone height), resulting in a 'shifted' distance map
		(half of the cloud was 'ignored')
	* DXF export
		- the I/O filter was mistakenly exporting the vertices of polylines and meshes as separate clouds
		- the I/O filter was not exporting the shifted point clouds at the right location
	* Render to file:
		- when the 'draw rounded points' option was enabled, pixel transparency could cause a strange effect when exported to PNG images
	* Octree rendering:
		- the 'Cube' mode was not functional
		- the 'Point' mode with normals was not functional