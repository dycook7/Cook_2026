// Enter model name
modelName = "P98_12mo_";
// Directory will probably be the same, but double check
dir = "E:/Processed_vein_images/P98_12mo_processed/";

// Define the number of pressures (usually from 12-15)
numberofpressures = getNumber("Input number of pressures", 14);
pressures = newArray(numberofpressures);
for (i = 0; i < numberofpressures; i++) {
    pressures[i] = (i + 1) * 100;
}

// Define locations
regions = newArray("Prebend", "Bend", "Postbend");
sections = newArray("Inner", "Outer");
prebendPositions = newArray("Pos1", "Pos2", "Pos3", "Pos4");
postbendPositions = newArray("Pos1", "Pos2", "Pos3", "Pos4");

bendCount = getNumber("Enter the number of positions for Bend region:", 4);
bendPositions = newArray(bendCount);
for (i = 0; i < bendCount; i++) {
    bendPositions[i] = "Pos" + (i + 1);
}

index = 0;
totalPositions = prebendPositions.length + bendCount + postbendPositions.length;
ROI = newArray(totalPositions * sections.length);
ROI_name = newArray(totalPositions * sections.length);

for (r = 0; r < regions.length; r++) {
    if (regions[r] == "Prebend") {
        positions = prebendPositions;
    } else if (regions[r] == "Bend") {
        positions = bendPositions;
    } else {
        positions = postbendPositions;
    }

    for (s = 0; s < sections.length; s++) {
        for (p = 0; p < positions.length; p++) {
            ROI[index] = "/" + regions[r] + "/" + sections[s] + "/" + positions[p] + "/";
            ROI_name[index] = regions[r] + "_" + sections[s] + "_" + positions[p] + "_";
            index++;
        }
    }
}

// Input starting location
startRegion = getString("Input start region (Prebend, Bend, Postbend)", "Prebend");
startSection = getString("Input start section (Inner, Outer)", "Inner");
startPosition = getString("Input start position (Pos1, Pos2, Pos3, Pos4)", "Pos1");
startPressure = parseInt(getString("Input start pressure (multiple of 100)", "100"));

// Find starting index in ROI array
startIndex = -1;
for (i = 0; i < ROI.length; i++) {
    if (ROI_name[i] == startRegion + "_" + startSection + "_" + startPosition + "_") {
        startIndex = i;
        break;
    }
}

// Ensure the starting point exists
if (startIndex == -1) {
    print("Starting point not found, using default start.");
    startIndex = 0;
}

// Open files and perform operations
for (i = startIndex; i < ROI.length; i++) {
    baseLocation = ROI[i];
    locationName = ROI_name[i];
    
    for (d = 0; d < pressures.length; d++) {
        if (i == startIndex && pressures[d] < startPressure) {
            continue;  // Skip pressures less than startPressure for the start location
        }
      
        // Establish file path for specific sub-folder and open files as stack
        filepath = dir + baseLocation + pressures[d] + "mbar";
        streamlines = filepath + "/" + modelName + locationName + pressures[d] + "mbar_streamlines.tif";
        open(streamlines);
        contour = filepath + "/" + modelName + locationName + pressures[d] + "mbar_contour.tif";
        open(contour); 
        
        streamlines = modelName + locationName + pressures[d] + "mbar_streamlines.tif";
        contour = modelName + locationName + pressures[d] + "mbar_contour.tif";
        
        selectWindow(contour);
        run("Clear Outside");
        run("Invert LUTs");
        
        selectWindow(streamlines);
        runMacro("C:/Users/dycook/OneDrive - The University of Chicago/Documents/ImageJMacros/Shanbhag_threshold.ijm");
        run("Make Binary", "method=Default background=Dark");
        run("Analyze Particles...", "size=100-5000 pixel circularity=0.3-0.4 show=Masks exclude stack");
        selectWindow(streamlines);
        runMacro("C:/Users/dycook/OneDrive - The University of Chicago/Documents/ImageJMacros/AddBlankSlice.ijm");
        selectWindow(contour);
        run("Invert LUTs");
        
        imageCalculator("Add stack", streamlines, contour);
        
        selectWindow(contour);
        run("Save");
        selectWindow(streamlines);
        run("Save");
        close("*");
    }    
}