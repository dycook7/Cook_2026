# This code is designed to display WSS or angle data onto vein model .vtp files
# .vtp files for models used in Cook et al. are provided

import pyvista as pv
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib as plt
from matplotlib.colors import LinearSegmentedColormap


############################################################

# Code Setup
## Choose model type
toyModel = True
## Choose data to display
### Either "Angle" or "WSS"
input_data_type = "Angle"

############################################################
pv.global_theme.allow_empty_mesh = True

# List all model names
toyModelNames = ("3mm_Toy", "6mm_Toy", "9mm_Toy")
patientModelNames = ("P93_3mo", "P93_12mo", "P96_3mo", "P96_12mo", "P98_3mo", "P98_12mo", "P104_3mo", "P104_12mo")

if toyModel:
    modelNames = toyModelNames
else:
    modelNames = patientModelNames

## Do you want an interactive plot? 
### For this code, no, you don't
interactive = False

# Creating a custom color map for later scale bars
colors = ["lightskyblue", "midnightblue"]  
custom_cmap_linear = LinearSegmentedColormap.from_list(
    name='my_linear_cmap',
    colors=colors
)
# Optional: Use a matplotlib predefined color map 
color_map = custom_cmap_linear

# Managing file paths
if input_data_type == "Angle":
    data_type = 'Angle_mean'
    scalars = 'Streamline Angle (°)'
    folder = "Angle"

if input_data_type == "WSS":
    data_type = 'WSS_mean'
    scalars = "WSS (mPa)"
    folder = "WSS"


for modelName in modelNames:    
    if modelName == "3mm_Toy":
        pressureMax = 1601
        
    if modelName == "6mm_Toy":
        pressureMax = 1201

    if modelName == "9mm_Toy":
        pressureMax = 1301
        
    if modelName == "P93_3mo":
        pressureMax = 1401

    if modelName == "P93_12mo":
        pressureMax = 1401
        
    if modelName == "P96_3mo":
        pressureMax = 1301
        
    if modelName == "P96_12mo":
        pressureMax = 1401

    if modelName == "P98_3mo":
        pressureMax = 1401
        
    if modelName == "P98_12mo":
        pressureMax = 1401

    if modelName == "P104_3mo":
        pressureMax = 1501
        
    if modelName == "P104_12mo":
        pressureMax = 1501
    
    # Define file paths for model, inner wall, and outer wall
    pressure_range = range(100, pressureMax, 100)
    base_path = r"C:\Users\dycook\Documents\Python\VMTK\Models"
    vessel     = pv.read(f"{base_path}/{modelName}/{modelName}_vessel.vtp")
    inner_wall = pv.read(f"{base_path}/{modelName}/{modelName}_inner.vtp")
    outer_wall = pv.read(f"{base_path}/{modelName}/{modelName}_outer.vtp")

    # Read in data
    if toyModel:
        df = pd.read_excel(r"C:\Users\dycook\Documents\Python\toy_models_python_combined_summary.xlsx")
    else:
        df = pd.read_excel(r"C:\Users\dycook\Documents\Python\patient_models_python_combined_summary.xlsx")

    # Define function to get inner/outer wall values
    ## Also calculates lower and upper deciles of WSS data
    def get_angle_mean_arrays_allregions(df, model_name, pressure, data_type, region_order=["Prebend", "Bend", "Postbend"]):
        inners = []
        outers = []

        
        for region in region_order:
            df_model = df[(df['ModelName'] == model_name) & 
                        (df['Region'] == region) &
                        (df['Pressure'] == pressure)]
            # Sort to keep a predictable anatomical order
            df_model = df_model.sort_values(by=['Wall', 'Position'])
            inner = df_model[df_model['Wall'] == "Inner"][data_type].to_numpy()
            outer = df_model[df_model['Wall'] == "Outer"][data_type].to_numpy()
            inners.extend(inner)
            outers.extend(outer)
        
        df_2 = df[df['ModelName'] == model_name]
        clean_subset = df_2[data_type].dropna()
        lower_decile = np.quantile(clean_subset, 0.1)
        upper_decile = np.quantile(clean_subset, 0.9)
        

        return np.array(inners), np.array(outers), lower_decile, upper_decile

    # Define function that segments models into bins, then assigns scalar value
    def assign_scalar_by_arclength(mesh, values):
        """
        Assigns scalar values to mesh points based on their normalized arclength,
        split into len(values) equal bins.
        """
        points = mesh.points
        # Use PCA to determine the vessel's main axis (don't overthink this)
        pca = PCA(n_components=1)
        distances = pca.fit_transform(points)[:, 0]
        # Normalize distances to [0, 1]
        norm_d = (distances - distances.min()) / (distances.max() - distances.min())
        # Bin points into len(values) segments
        bins = np.linspace(0, 1, len(values)+1)
        bin_idx = np.digitize(norm_d, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(values)-1)
        # Assign the flow value from your supplied array
        return np.array([values[j] for j in bin_idx])


    def assign_scalar_by_arclength_middle_band(mesh, values, band_frac=0.40):
        """
        Assigns scalar values to mesh points based on their normalized arclength,
        but only the middle band_frac of each bin gets the value. Rest get np.nan.

        Parameters:
            mesh: pyvista mesh object
            values: array of values, one per bin
            band_frac: fractional width of the middle band in each bin (0 < band_frac < 1)
        """
        points = mesh.points
        pca = PCA(n_components=1)
        distances = pca.fit_transform(points)[:, 0]
        norm_d = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        bins = np.linspace(0, 1, len(values)+1)

        center_width = band_frac
        half_side = (1 - center_width) / 2

        bin_idx = np.digitize(norm_d, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(values)-1)

        scalars = np.full(len(norm_d), np.nan)
        for i in range(len(values)):
            start = bins[i]
            end = bins[i+1]
            mid_start = start + half_side * (end - start)
            mid_end   = end - half_side * (end - start)
            mask = (norm_d >= mid_start) & (norm_d < mid_end) & (bin_idx == i)
            scalars[mask] = values[i]
        return scalars


    def split_mesh_by_nan(mesh):
        """
        Splits a mesh into two separate meshes based on NaN values
        in the active scalar array.

        Args:
            mesh (pyvista.DataSet): The input PyVista mesh with an
                                    active scalar array.

        Returns:
            tuple[pyvista.DataSet, pyvista.DataSet]: A tuple containing two meshes:
                - nan_mesh: The mesh containing only cells or points where
                            the active scalar is NaN.
                - valid_mesh: The mesh containing only cells or points where
                            the active scalar is not NaN.
        """
        # Get the name of the active scalar array
        scalar_name = mesh.active_scalars_name

        if scalar_name is None:
            raise ValueError("No active scalar array found on the mesh.")

        # Determine if the active scalar is on the points or cells
        if scalar_name in mesh.point_data:
            # Use threshold filter for point data
            nan_mesh = mesh.threshold([-np.inf, np.inf], invert=True, scalars=scalar_name, preference='point')
            valid_mesh = mesh.threshold(scalars=scalar_name, preference='point')

        elif scalar_name in mesh.cell_data:
            # Use masks and extract_cells for cell data
            scalars = mesh.cell_data[scalar_name]
            is_nan_mask = np.isnan(scalars)
            is_valid_mask = ~is_nan_mask

            nan_cell_indices = np.where(is_nan_mask)
            valid_cell_indices = np.where(is_valid_mask)

            nan_mesh = mesh.extract_cells(nan_cell_indices)
            valid_mesh = mesh.extract_cells(valid_cell_indices)

        else:
            # Fallback in case of unexpected scalar location
            raise RuntimeError(f"Unexpected location for scalar array '{scalar_name}'.")

        return nan_mesh, valid_mesh



    # For loop to iterate along pressure range in model
    for pressure in pressure_range:
        # Obtain inner/outer values, lower/upper decile
        inner_values, outer_values, lower_decile, upper_decile = get_angle_mean_arrays_allregions(df, modelName, pressure, data_type=data_type)

        # Assign those flow scalars based on position along wall
        inner_wall.point_data[scalars] = assign_scalar_by_arclength_middle_band(inner_wall, inner_values)
        outer_wall.point_data[scalars] = assign_scalar_by_arclength_middle_band(outer_wall, outer_values)
        
        inner_nans, inner_real = split_mesh_by_nan(inner_wall)
        outer_nans, outer_real = split_mesh_by_nan(outer_wall)

        # Conditional plots based on chosen data type
        if input_data_type == "WSS":
            minval = float(lower_decile)
            maxval = float(upper_decile)

            if interactive: 
                window_size = [1200, 800]
                title = modelName + str(pressure) + "mbar"
                
                bar_args = dict(
                    title_font_size=40,
                    label_font_size=34,
                    fmt="%.1f",
                    n_labels=3, 
                    font_family="arial",
                    above_label="Upper Decile",
                    below_label="Lower Decile"
                )
                
                p = pv.Plotter(off_screen=False, window_size=window_size)
                p.add_mesh(vessel, color='white', opacity=1, lighting=True, point_size=1)
                p.add_mesh(inner_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(outer_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(inner_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="red", above_color="mediumseagreen", 
                    nan_color="#A1A1A3", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                p.add_mesh(outer_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="red", above_color="mediumseagreen", 
                    nan_color="#DFDEDC", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                
                p.set_background('white')
                p.view_yx()
                p.add_text(title, font_size=20, color='black', position='upper_left')
                p.add_axes()
                p.show()
            else:
                window_size = [2400, 1800]
                title = modelName + str(pressure) + "mbar"
                
                bar_args = dict(
                    title_font_size=40,
                    label_font_size=34,
                    fmt="%.1f",
                    n_labels=3, 
                    font_family="arial",
                    above_label="Upper Decile",
                    below_label="Lower Decile"
                )
                
                
                p = pv.Plotter(off_screen=True, window_size=window_size)
                p.add_mesh(vessel, color='white', opacity=1, lighting=True, point_size=1)
                p.add_mesh(inner_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(outer_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(inner_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="red", above_color="mediumseagreen", 
                    nan_color="#A1A1A3", show_scalar_bar=True, lighting=False, smooth_shading=False, show_edges=False, scalar_bar_args=bar_args)
                p.add_mesh(outer_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="red", above_color="mediumseagreen", 
                    nan_color="#DFDEDC", show_scalar_bar=True, lighting=False, smooth_shading=False, show_edges=False, scalar_bar_args=bar_args)
                p.view_yx()  # y axis to right, x up
                p.add_text(title, font_size=20, color='black', position='upper_left')
                
                p.reset_camera()
                p.show(auto_close=False)  # Off-screen: this only triggers draw, doesn't pop up a GUI
                p.screenshot(f"Renders/{folder}/{modelName}/{modelName}_{pressure}mbar_{folder}.png", transparent_background=True)  # Now this works because the scene exists!
                p.close()
            
        else:  
            minval = 5
            maxval = 45
            
            if interactive: 
                window_size = [1200, 800]
                title = modelName + str(pressure) + "mbar"
                
                bar_args = dict(
                    title_font_size=40,
                    label_font_size=34,
                    fmt="%.1f",
                    n_labels=3, 
                    font_family="arial",
                    below_label="<5°",
                    above_label=">45°"
                )
                
                p = pv.Plotter(off_screen=False, window_size=window_size)
                p.add_mesh(vessel, color='white', opacity=1, lighting=True, point_size=1)
                p.add_mesh(inner_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(outer_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(inner_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="mediumseagreen", above_color="red", 
                    nan_color="#A1A1A3", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                p.add_mesh(outer_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="mediumseagreen", above_color="red", 
                    nan_color="#DFDEDC", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                
                p.set_background('white')
                p.view_yx()
                p.add_text(title, font_size=20, color='black', position='upper_left')
                p.add_axes()
                p.show()
            else:
                window_size = [2400, 1800]
                title = modelName + str(pressure) + "mbar"
                
                bar_args = dict(
                    title_font_size=40,
                    label_font_size=34,
                    fmt="%.1f",
                    n_labels=3, 
                    font_family="arial",
                    below_label="<5°",
                    above_label=">45°"
                )
                
                p = pv.Plotter(off_screen=True, window_size=window_size)
                p.add_mesh(vessel, color='white', opacity=1, lighting=True, point_size=1)
                p.add_mesh(inner_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(outer_nans, color="white", opacity=1, lighting=True)
                p.add_mesh(inner_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="mediumseagreen", above_color="red", 
                    nan_color="#A1A1A3", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                p.add_mesh(outer_real, scalars=scalars, cmap=color_map, clim=[minval, maxval], below_color="mediumseagreen", above_color="red", 
                    nan_color="#DFDEDC", show_scalar_bar=True, lighting=False, smooth_shading=True, show_edges=False, scalar_bar_args=bar_args)
                
                p.view_yx()  # y axis to right, x up
                p.add_text(title, font_size=20, color='black', position='upper_left')
                
                p.reset_camera()

                p.show(auto_close=False)  # Off-screen: this only triggers draw, doesn't pop up a GUI
                p.screenshot(f"Renders/{folder}/{modelName}/{modelName}_{pressure}mbar_{folder}.png", transparent_background=True)  # Now this works because the scene exists!
                p.close()