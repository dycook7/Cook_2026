from singleVideoProcessing import *
from ViolinPlotFun import *
import pandas as pd
from os.path import exists
import numpy as np
import os

# This code is designed to take an input of an Excel spreadsheet containing metadata
# This spreadsheet provides:
## 1) Parameter values needed for calculations
## 2) File paths to raw data
# An example spreadsheet has been provided

file_path = r"C:/path_to_your_spreadsheet.xlsx"

meta = pd.read_excel(file_path)
meta = meta.reset_index()  # make sure indexes pair with number of rows

meta_sub = meta

# If you need to check what you're working with, uncomment the line below
# print(meta_sub)

basepath = r"C:/base_path_to_your_data_files"
ForceRecomputeWSS = True
ForceRecomputeV = False

for index, row in meta_sub.iterrows():
    if pd.notna(row["TIFF File Name"]):
        folder = basepath + row['Folder Path'] + "/"
        filename = row['TIFF File Name']
        modelName = row['Model']
        viscosity = row['Viscosity (cP)']
        exp_time = row['Exposure Time (s)']
        mag = row['Magnification (X)']
        flowRate = row['Flow Rate (mL/min)']
        region = row['Region']
        wall = row['Wall']
        position = row['Position']
        pixel2mm = row['Pixel to mm']
        density = row['Density (g/mL)']
        bmf = row['BMF (%)']
        
        if not os.path.exists(folder + filename):
            print(folder + filename + " does not exist! Check meta.csv")
            continue
        
        if filename != "" and os.path.exists(folder + filename):
            fname2folder = filename.split(".TIF")[0]
            fname2folder = fname2folder.split(".tif")[0]
            outputfolder = folder + "/" + fname2folder + "/"
            # print(outputfolder)
            
            if exists(outputfolder + "wss.pkl") and not ForceRecomputeWSS:
                print(filename + ":processed")
            else:
                print(filename + ":NotYet")
                # print(exp_time, viscosity, pixel2mm)
                singleVideoProcessing(folder, filename, filename, exp_time, viscosity, pixel2mm)
            
            if exists(outputfolder + "wss_per_frame.csv") and not ForceRecomputeWSS:
                print(filename + ":csv generated")
            else:
                wss = pickle.load(open(outputfolder + "wss.pkl", "rb"))
                with open(outputfolder + "wss_per_frame.csv", 'w') as f:
                    for frame in range(len(wss[0])):
                        for key in wss[0][frame].keys():
                            f.write("%s, %f, %d\n" % (key, wss[0][frame][key], frame))
                
                with open(outputfolder + "wss_collapsed.csv", 'w') as f:
                    for key in wss[1].keys():
                        f.write("%s, %f\n" % (key, wss[1][key]))
                
                lines = pickle.load(open(outputfolder + "linesAllFrame.pkl", "rb"))
                
                with open(outputfolder + "Line_per_frame.csv", 'w') as f:
                    with open(outputfolder + "Velocity_per_frame.csv", 'w') as f1:
                        for frame in range(len(lines)):
                            if lines[frame] != []:
                                clean_data = []
                                for element in lines[frame]:
                                    if isinstance(element, list):
                                        clean_data.append(element[4])
                                    else:
                                        # print(f"Unexpected element: {element}")
                                        pass
                                
                                try:
                                    frVelo = np.array(clean_data)
                                    # print(f"frVelo: {frVelo}")
                                except ValueError as e:
                                    print(f"Error: {e}")
                                
                                f1.write("%d,%f,%f,%f,%f\n" % (frame, np.max(frVelo), np.min(frVelo), np.mean(frVelo), np.median(frVelo)))
                                for l in lines[frame]:
                                    f.write("%f,%f,%f,%f,%d,%d,%f,%d\n" % (l[0][0], l[0][1], l[1][0], l[1][1], l[2], l[3], l[4], frame))
                                    
            if exists(outputfolder + "wss_violinPlot.png") and not ForceRecomputeV:
                print("Violin exists")
            else:
                print("Computing Violin")
                try:
                    ViolinPlot(outputfolder, exp_time, viscosity, pixel2mm)
                except Exception as e:
                    print(f"Error processing folder {outputfolder}: {e}")