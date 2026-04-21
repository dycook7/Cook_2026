# This file defines the singleVideoProcessing function
# This function takes a .tif stack of ImageJ-processed images and metadata to compute WSS and streamline angle

import cv2
import numpy as np
import tifffile
from config import *    # written by us
from SubFun import SubFun  # written by us
import pickle
import sys
import os
import matplotlib.pyplot as plt

#### Setting default properties for text to be printed on the image
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_OLIVE = (0, 128, 128)

font = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.8
fontColor = COLOR_GREEN
lineType = 3


def singleVideoProcessing(folder, fname, contourfile, time, viscosity, pixel2mm):
    fname2folder = fname.split(".TIF")[0]
    fname2folder = fname2folder.split(".tif")[0]
    outputfolder = folder + "/" + fname2folder + "/"

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder, exist_ok=False)
    if not os.path.exists(outputfolder + "FilteredLines"):
        os.makedirs(outputfolder + "FilteredLines", exist_ok=False)
    if not os.path.exists(outputfolder + "ProjectLines"):
        os.makedirs(outputfolder + "ProjectLines", exist_ok=False)
    if not os.path.exists(outputfolder + "individual"):
        os.makedirs(outputfolder + "individual", exist_ok=False)
    if not os.path.exists(outputfolder + "origin"):
        os.makedirs(outputfolder + "origin", exist_ok=False)

    ########### Get wall pixels
    outline = cv2.imread(folder + contourfile)
    wall, edge = SubFun.GetWallFromImage(outline, True, outputfolder)

    edgeColor_lines = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    a = tifffile.imread(folder + fname)
    wall2wss = dict()

    ### Loop through all frames
    filImage = list()
    projImage = list()
    line_frame_all = list()
    for framei in range(len(a)):
        gray = a[framei]
        cv2.imwrite(outputfolder + "origin/original_" + str(framei) + ".png", gray)
        ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(gray)
        ## Filter CCs if too small < 75 pixels or too big > 500 pixels
        ## OPTIONS: change 75 and 500 to other numbers
        validlabel = np.where((stats2[:, 4] >= 75) & (stats2[:, 4] < 9000))
        labels2_filt = np.zeros((labels2.shape[0], labels2.shape[1]))
        for cc in range(len(validlabel[0])):
            labels2_filt[labels2 == validlabel[0][cc]] = cc + 1
        ## Get lines from the CCs
        ## OPTIONS:
        # box ratio control: ratioUB=10, ratioLB=3,
        # box area control:  areaUB=2000, areaLB=80,
        # Angle control mean+/-std: angleM=None, angleStd=20,
        # Line length control: pixelLB=75, pixelUB=150
        # allline_stat2: [startpoint[0], startpoint[1], endpoint[0], endpoint[1],
        #                length_rect, angle, bwidth, bheight, labelindex]
        newCanvas = cv2.cvtColor(a[framei], cv2.COLOR_GRAY2BGR).copy()
        filteredImage2, unfiltered_lines2, filteredLines2, unfilteredImage2, allline_stat2, filline_stat2 = SubFun.detectLineFromCCs(
            newCanvas, labels2_filt.astype('int'), font, fontScale,
            fontColor, lineType, ratioLB=3, ratioUB=100, areaUB=9000, areaLB=0, pixelLB=15, pixelUB=500, lineWidthUB=50)
        cv2.imwrite(outputfolder + "FilteredLines/Fil_lines" + str(framei) + ".png", filteredImage2)
        filImage.append(filteredImage2.copy())
        ## Filtered lines renamed as linesFrame
        linesFrame = filteredLines2
        maxD = cv2.norm(edge.shape)
        edgeColor = cv2.cvtColor(a[framei], cv2.COLOR_GRAY2BGR)
        ## Loop through each line, project it to the wall contour
        ## For each pixel in the wall save the list of lines projected from all frames

        for i in range(len(linesFrame)):
            [startpoint, endpoint, length, angle] = linesFrame[i]
            velocity = (length / pixel2mm) / time
            linesFrame[i].append(velocity)
            boxmin, boxmax, min_d, streamline_angle = SubFun.FindContourProjectionBox([startpoint, endpoint], [edge.shape[0] - 1, edge.shape[1] - 1], wall)
            if min_d == 0 and boxmin[0] == 0 and boxmax[0] == 0:
                continue
            x_diff = boxmax[0] - boxmin[0]
            y_diff = boxmax[1] - boxmin[1]
            if x_diff < 5:
                boxmin = (boxmin[0] - 10, boxmin[1])
                boxmax = (boxmax[0] + 10, boxmax[1])
            if y_diff < 5:
                boxmin = (boxmin[0], boxmin[1] - 10)
                boxmax = (boxmax[0], boxmax[1] + 10)
            cv2.rectangle(edgeColor, boxmin, boxmax, COLOR_GREEN)
            midpoint = (int((boxmin[0] + boxmax[0]) / 2), int((boxmin[1] + boxmax[1]) / 2))
            cv2.line(edgeColor, (int(startpoint[0]), int(startpoint[1])), midpoint, COLOR_CYAN, thickness=lineType)
            cv2.line(edgeColor, (int(endpoint[0]), int(endpoint[1])), midpoint, COLOR_CYAN, thickness=lineType)
            cv2.line(edgeColor, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])), COLOR_YELLOW, thickness=lineType)
            for w in wall:
                if w[0] <= boxmax[0] and w[0] >= boxmin[0] and w[1] <= boxmax[1] and w[1] >= boxmin[1]:
                    cv2.circle(edgeColor, w, radius=lineType, color=COLOR_OLIVE)
                    if tuple(w) not in wall2wss:
                        wall2wss[tuple(w)] = list()
                    wss = float((length / time) / min_d * viscosity) #mPa
                    # wss = float(wss * 0.01) #convert to dyn/cm2
                    wall2wss[tuple(w)].append([framei, i, startpoint[0], startpoint[1], endpoint[0], endpoint[1], length, angle, min_d, wss, streamline_angle])

        cv2.imwrite(outputfolder + "ProjectLines/" + str(framei) + "edgeProjectedLines.png", edgeColor)
        projImage.append(edgeColor.copy())
        line_frame_all.append(linesFrame)

    with open(outputfolder + "linesAllFrame.pkl", 'wb') as f:
        pickle.dump(line_frame_all, f)

    wssByEdgePoint_split = dict()
    for framei in range(len(a)):
        wssByEdgePoint_split[framei] = dict()
    wssByEdgePointAll = dict()
    listwss_all = list()
    listwss_ave_per_frame = list()
    listwss_ave_global = list()
    streamline_angles_all = list()  # Store streamline angles for statistics

    for w in wall2wss.keys():
        listwss = np.array(wall2wss[w])
        listwss_all = listwss_all + wall2wss[w]
        streamline_angles_all.extend(listwss[:, -1].tolist())  # Collect streamline angles
        wssMean = np.mean(listwss[:, 9])
        listwss_ave_global.append(wssMean)
        wssByEdgePointAll[w] = wssMean
        for framei in range(len(a)):
            wssMean = np.mean(listwss[listwss[:, 0] == framei, 9])
            if not np.isnan(wssMean):
                wssByEdgePoint_split[framei][w] = wssMean
                listwss_ave_per_frame.append(wssMean)

    with open(outputfolder + "wss.pkl", 'wb') as f:
        pickle.dump([wssByEdgePoint_split, wssByEdgePointAll], f)

    
    if len(listwss_all) == 0:
        print("listwss_all is empty. Skipping processing for this file.")
        return #return control back to the calling code
    
    listwss_all = np.array(list(listwss_all))
    listwss_ave_per_frame = np.array(listwss_ave_per_frame)
    listwss_ave_global = np.array(listwss_ave_global)
    streamline_angles = np.array(streamline_angles_all)  # Convert to NumPy array

    with open(outputfolder + 'Summary.csv', 'w') as f:
        print("Type,min,mean,median,max,std", file=f)
        print("Global," + ",".join([
            "{:.2f}".format(np.min(listwss_ave_global)),
            "{:.2f}".format(np.mean(listwss_ave_global)),
            "{:.2f}".format(np.median(listwss_ave_global)),
            "{:.2f}".format(np.max(listwss_ave_global)),
            "{:.2f}".format(np.std(listwss_ave_global))]), file=f)
        print("Per Frame," + ",".join([
            "{:.2f}".format(np.min(listwss_ave_per_frame)),
            "{:.2f}".format(np.mean(listwss_ave_per_frame)),
            "{:.2f}".format(np.median(listwss_ave_per_frame)),
            "{:.2f}".format(np.max(listwss_ave_per_frame)),
            "{:.2f}".format(np.std(listwss_ave_per_frame))]), file=f)
        print("Raw (Per streamline & wall pixel)," + ",".join([
            "{:.2f}".format(np.min(listwss_all)),
            "{:.2f}".format(np.mean(listwss_all)),
            "{:.2f}".format(np.median(listwss_all)),
            "{:.2f}".format(np.max(listwss_all)),
            "{:.2f}".format(np.std(listwss_all))]), file=f)
        print("Streamline angle," + ",".join([
            "{:.2f}".format(np.min(streamline_angles)),
            "{:.2f}".format(np.mean(streamline_angles)),
            "{:.2f}".format(np.median(streamline_angles)),
            "{:.2f}".format(np.max(streamline_angles)),
            "{:.2f}".format(np.std(streamline_angles))]), file=f)
        print("Full coverage," + str(len(wssByEdgePointAll) / len(wall)), file=f)

    minwss_per_frame = np.min(listwss_ave_per_frame)
    maxwss_per_frame = np.max(listwss_ave_per_frame)
    minwss_collapsed_global = np.min(listwss_ave_global)
    maxwss_collapsed_global = np.max(listwss_ave_global)

    edgeColorWss = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    imgray = np.array(range(256)).astype('uint8')
    imcolor = np.flip(cv2.applyColorMap(imgray, cv2.COLORMAP_AUTUMN), axis=0)  ## yellow to red scale
    from math import log
    for e in wall:
        if tuple(e) in wssByEdgePointAll:
            current_wss = wssByEdgePointAll[tuple(e)]
            if minwss_collapsed_global == maxwss_collapsed_global:
                break
            wall_color = np.clip((log(current_wss) - log(minwss_collapsed_global)) * 255 / (log(maxwss_collapsed_global) - log(minwss_collapsed_global)), 0, 255).astype('uint8')
            color = (int(imcolor[wall_color][0][0]), int(imcolor[wall_color][0][1]), int(imcolor[wall_color][0][2]))
            cv2.circle(edgeColorWss, tuple(e), 2, color)
    cv2.imwrite(outputfolder + "colorEdges_results_thick_all.png", edgeColorWss)

    coverage_all = list()
    for i in range(len(a)):
        edgeColorWss = cv2.cvtColor(a[i], cv2.COLOR_GRAY2BGR)
        edgeColorWss = np.clip(edgeColorWss * 0.5, 0, 255)
        imgray = np.array(range(256)).astype('uint8')
        imcolor = np.flip(cv2.applyColorMap(imgray, cv2.COLORMAP_AUTUMN), axis=0)  ## blue to purple scale
        num_wall = len(wall)
        coverage = 0
        for e in wall:
            if tuple(e) in wssByEdgePoint_split[i]:
                coverage = coverage + 1
                current_wss = wssByEdgePoint_split[i][tuple(e)]
                if maxwss_per_frame == minwss_per_frame:
                    break
                wall_color = np.clip((log(current_wss) - log(minwss_per_frame)) * 255 / (log(maxwss_per_frame) - log(minwss_per_frame)), 0, 255).astype('uint8')
                color = (int(imcolor[wall_color][0][0]), int(imcolor[wall_color][0][1]), int(imcolor[wall_color][0][2]))
                cv2.circle(edgeColorWss, tuple(e), 2, color)
                cv2.circle(filImage[i], tuple(e), 2, color)
                cv2.circle(projImage[i], tuple(e), 2, color)
        coverage_all.append(coverage / num_wall)
        cv2.imwrite(outputfolder + "individual/colorEdges_origImage_thick_" + str(i) + ".png", edgeColorWss)
        cv2.imwrite(outputfolder + "individual/colorEdges_FilLinesImage_thick_" + str(i) + ".png", filImage[i])
        cv2.imwrite(outputfolder + "individual/colorEdges_ProjLinesImage_thick_" + str(i) + ".png", projImage[i])

    try:
        wss_values = np.array(list(wssByEdgePointAll.values())).astype('float')
        if len(wss_values) == 0:
            raise ValueError("WSS values array is empty")

        print("WSS Values Array Shape:", wss_values.shape)
        print("WSS Values Array Contents:", wss_values[:10], "...")  # First 10 values

        fig, ax = plt.subplots()
        ax.hist(wss_values, bins=100)
        ax.set_xlabel("WSS (mPa)")
        ax.set_ylabel("Frequency(Counts)")

        # Ensure the output folder path exists
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
            print(f"Output folder created: {outputfolder}")

        full_path = os.path.join(outputfolder, "WSS_distribution.png")
        plt.savefig(full_path)
        plt.close()
        print(f"Distribution plot saved successfully at {full_path}")
    except ValueError as e:
        print(f"Error creating histogram for wssByEdgePointAll: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    try:
        fig = plt.figure()
        plt.bar(range(1, len(a) + 1), np.array(coverage_all) * 100)
        plt.xlabel("Frame Index")
        plt.ylabel("Wall Coverage(%)")
        plt.savefig(outputfolder + "coverage_Bar.png")
        plt.close()
    except ValueError as e:
        print(f"Error creating coverage bar plot: {e}")
    # Optionally, add a final print to make sure it completes
    print("Processing completed for:", fname)