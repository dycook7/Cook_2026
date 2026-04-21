# This script defines functions to create violin plots of WSS data

import numpy as np
import pickle
#import seaborn as sns
import matplotlib.pyplot as plt
import sys


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, max(vals))
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, min(vals), q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('')


def ViolinPlot(folder, exp_time, viscosity, pixel2mm):
    # Load WSS from the saved pkl file
    with open(folder + "wss.pkl", 'rb') as f:
        [wssByEdgePoint_split, wssByEdgePointAll] = pickle.load(f)

    with open(folder + "linesAllFrame.pkl", 'rb') as f:
        line_frame_all = pickle.load(f)

    # Plot
    temp = {
        'WSS': [],
        'Frame': {},
        'Velocity': []
    }
    allV = []
    totallines = 0
    maxwss = float('-inf')
    minwss = float('inf')
    maxV = float('-inf')
    minV = float('inf')

    for k, wss_values in wssByEdgePoint_split.items():
        if len(wss_values) > 0:
            wsslist = np.array(list(wss_values.values()))
            if line_frame_all.get(k):
                velocitylist = np.array(line_frame_all[k], dtype="object")[:, 2].astype("float") / (pixel2mm * exp_time)
                for p in range(len(line_frame_all[k])):
                    line_frame_all[k][p][4] = velocitylist[p]

                allV += velocitylist.tolist()  # Convert numpy array to list before appending
                temp['Velocity'].append(velocitylist)
                temp['WSS'].append(wsslist)

                maxwss = max(maxwss, np.max(wsslist))
                minwss = min(minwss, np.min(wsslist))
                maxV = max(maxV, np.max(velocitylist))
                minV = min(minV, np.min(velocitylist))

                linesPerFrame = len(line_frame_all[k])
                temp['Frame'][k] = linesPerFrame
                totallines += linesPerFrame

    with open(folder + "linesAllFrame.pkl", 'wb') as f:
        pickle.dump(line_frame_all, f)

    temp['WSS'].append(np.array(list(wssByEdgePointAll.values())))
    temp['Frame']["All"] = totallines
    temp['Velocity'].append(np.array(allV))

    data = temp['WSS']

    fig, (ax, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(int(8 * len(data) / 35), 12), sharex=True, gridspec_kw={'height_ratios': [5, 3, 2]})
    part = ax.violinplot(data, range(1, len(data) + 1), showmeans=False, showmedians=False, showextrema=False)

    quartile1 = []
    medians = []
    quartile3 = []
    for i in range(len(data)):
        if len(data[i]) > 0:
            q1, med, q3 = np.percentile(data[i], [25, 50, 75])
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
        else:
            print(f"Warning: Empty dataset at index {i}. Skipping quartile and whisker calculations.")
            quartile1.append(float('nan'))
            medians.append(float('nan'))
            quartile3.append(float('nan'))

    whiskers = np.array([
        adjacent_values(np.sort(d), q1, q3)
        for d, q1, q3 in zip(data, quartile1, quartile3)
        if len(d) > 0
    ])
    if len(whiskers):
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    else:
        whiskersMin = whiskersMax = []

    inds = np.arange(1, len(medians) + 1)

    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    ax.set_ylim([minwss, maxwss])
    ax.set_ylabel("WSS value")

    plt.tight_layout()

    # Save the plot with appropriate error handling
    try:
        plt.savefig(folder + "wss_violinPlot.png", dpi=200)
    except Exception as e:
        print(f"Error saving figure at folder {folder}: {e}")
    finally:
        plt.close()

    return temp
    

def ViolinPlot_collective(folder_collect):
    # Load WSS from saved pkl file
    with open(folder_collect + "wss.pkl", 'rb') as f:
        wss_wall = pickle.load(f)

    with open(folder_collect + "linesAllFrame.pkl", 'rb') as f:
        line_wall = pickle.load(f)

    # plot
    temp = {}
    temp['WSS'] = list()
    temp['Velocity'] = list()
    allV = list()
    totallines = 0
    maxwss = 0
    minwss = 100
    maxV = 0
    minV = 100

    for k in wss_wall.keys():
        if len(wss_wall[k]) > 0:
            wsslist = np.array(wss_wall[k])
            velocitylist = np.array(line_wall[k], dtype="object")[:, 4].astype("float")
            allV = np.append(allV, velocitylist)
            temp['Velocity'].append(velocitylist)
            temp['WSS'].append(wsslist)

            if maxwss < max(wsslist):
                maxwss = max(wsslist)
            if minwss > min(wsslist):
                minwss = min(wsslist)
            if maxV < max(velocitylist):
                maxV = max(velocitylist)
            if minV > min(velocitylist):
                minV = min(velocitylist)

            totallines = totallines + len(line_wall[k])

    # plot
    data = temp['WSS']
    fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [4, 4]})

    part = ax.violinplot(
        data, range(1, len(data) + 1), showmeans=False, showmedians=False,
        showextrema=False)
    set_axis_style(ax, wss_wall.keys())

    quartile1 = list()
    medians = list()
    quartile3 = list()
    for i in range(len(data)):
        t1, t2, t3 = np.percentile(data[i], [25, 50, 75])
        quartile1.append(t1)
        medians.append(t2)
        quartile3.append(t3)

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])

    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)

    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    ax.set_ylim([minwss, maxwss])
    ax.set_ylabel("WSS value")

    part1 = ax1.violinplot(
        temp['Velocity'], range(1, len(temp['Velocity']) + 1), showmeans=False, showmedians=False,
        showextrema=False)

    for pc in part1['bodies']:
        pc.set_facecolor('pink')
        pc.set_edgecolor('pink')
        pc.set_alpha(1)

    quartile1 = list()
    medians = list()
    quartile3 = list()
    for i in range(len(temp['Velocity'])):
        t1, t2, t3 = np.percentile(temp['Velocity'][i], [25, 50, 75])
        quartile1.append(t1)
        medians.append(t2)
        quartile3.append(t3)

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(temp['Velocity'], quartile1, quartile3)])

    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)

    ax1.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2)
    ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    ax1.set_ylim([minV, maxV])
    ax1.set_ylabel("Velocity(mm/s)")
    ax1.set_xlabel("ROIs")

    set_axis_style(ax1, wss_wall.keys())

    plt.savefig(folder_collect + "wss_violinPlot.png", dpi=300)
    plt.close()