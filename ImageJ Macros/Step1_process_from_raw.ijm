// Choose Model
modelName = getString("Input Model Name", "6mm_Toy");
modelName = modelName + "_"
dir = getDirectory("Choose a Directory");

// Define the number of pressures (usually from 12-15)
numberofpressures = getNumber("Input number of pressures", 14);
pressures = newArray(numberofpressures);
for (i = 0; i < numberofpressures; i++) {
    pressures[i] = (i + 1) * 100;
}

//Define locations
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

// Open files and perform operations
for (i = 0; i < ROI.length; i++) {
	baseLocation = ROI[i];
	locationName = ROI_name[i];
	
		for (d = 0; d < pressures.length; d++) {
		    // Establish file path for specific sub-folder and open files as stack
		    filepath = dir + baseLocation + pressures[d] + "mbar";
		    File.openSequence(filepath);
		    close("*.nd*");
		    //Rename for easier manipulation
		    rename("baseImage");
		    // Initial processing steps
		    run("Enhance Contrast...", "saturated=5 normalize equalize process_all");
		    run("Subtract Background...", "rolling=50 stack");
		    // Create MAX intensity projection for vein wall
		    run("Z Project...", "projection=[Max Intensity]");
		    // Construct savepath for pressure-specific folder
		    savepath = "E:/WSS_processing" + baseLocation + pressures[d] + "mbar";
		    projectionNewName = modelName + locationName + pressures[d] + "mbar_contour";
		    rename(projectionNewName);
		    saveAs("Tiff", savepath + "/" + projectionNewName + ".tif");
		    close();
		    
		    //Create AVG intensity, subtract it, despeckle, save to same folder
			selectWindow("baseImage");
			run("Z Project...", "projection=[Average Intensity]");
			selectWindow("AVG_" + "baseImage");
			rename("average");
		    imageCalculator("Subtract stack", "baseImage", "average");
		    selectWindow("average"); close();
		    run("Despeckle", "stack");
			subtractedNewName = modelName + locationName + pressures[d] + "mbar_streamlines";
		    saveAs("Tiff", savepath + "/" + subtractedNewName + ".tif");
		    close("*");
		}
}	