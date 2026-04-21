# This file defines a class of functions used to calculate WSS and streamline angle using CV2

import numpy as np
import cv2
import matplotlib.cm
from cv2_rolling_ball import subtract_background_rolling_ball

# This code contains functions necessary to calculate WSS and streamline angle

COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
RED = (255, 0, 0)

from shapely.geometry import LineString, Point
from config import *

class SubFun:
    def checkPointInRect(point, rect):
        x = point[0]
        y = point[1]
        xmin = rect[0]
        xmax = rect[1]
        ymin = rect[2]
        ymax = rect[3]
        if x > xmin and x < xmax and y > ymin and y < ymax:
            return True
        else:
            return False

    def GetWallFromImage(outline, saveEdge=False, name="", top=1):
        edges = cv2.Canny(outline, 1000, 200)
        edges = cv2.GaussianBlur(edges, (5, 5), cv2.BORDER_DEFAULT)
        # cv2.imwrite(name, edges)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longestContour = 0
        contourIndex = 0
        for i in range(len(contours)):
            c = contours[i]
            if c.shape[0] > longestContour:
                longestContour = c.shape[0]
                wall = c
                contourIndex = i
        edge = np.zeros(outline.shape[0:2], dtype="uint8")
        cv2.drawContours(edge, contours, contourIndex, 255, 1)
        ### Save the edge image if needed
        if saveEdge:
            cv2.imwrite(name + "edge_draw.png", edge)
        edgeColor_lines = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        wall = np.squeeze(contours[contourIndex], 1)
        wall_noDup = {}
        for w in wall:
            if tuple(w) not in wall_noDup:
                wall_noDup[tuple(w)] = 1
        wall = list(wall_noDup.keys())
        import csv
        with open(name + 'wall.csv', mode='w') as wallfile:
            filewriter = csv.writer(wallfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerows(wall)
        return wall, edge

    def Get2WallFromImage(outline, saveEdge=False, name="", top=2):
        ret, thresh1 = cv2.threshold(outline, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(255 - thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        longestContour = 0
        contourIndex = 0
        contourLength = list()
        for i in range(len(contours)):
            c = contours[i]
            contourLength.append(c.shape[0])
        contourIndex = np.argsort(-np.array(contourLength))[0:top]

        for i in contourIndex:
            edge = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(edge, contours, i, COLOR_GREEN, 1)
            ### Save the edge image if needed
            if saveEdge:
                cv2.imwrite(name + "edge_draw" + str(i) + ".png", edge)
        wall_all = list()
        for i in contourIndex:
            wall = np.squeeze(contours[i], 1)
            wall_noDup = {}
            for w in wall:
                if tuple(w) not in wall_noDup:
                    wall_noDup[tuple(w)] = 1
            wall = list(wall_noDup.keys())
            import csv
            with open(name + 'wall' + str(i) + '.csv', mode='w') as wallfile:
                filewriter = csv.writer(wallfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerows(wall)
            wall_all.append(wall)
        return wall_all, edge

    def FindContourProjectionBox(line, maxpoint, contour):
        from shapely.geometry import LineString, Point
        import numpy as np
        import math
        
        a = line[0]
        b = line[1]

        xmin = 0
        xmax = maxpoint[0]
        ymin = 0
        ymax = maxpoint[1]

        # If the y-coordinates of points a and b are not the same
        if a[1] != b[1]:
            k = -(a[0] - b[0]) / (a[1] - b[1])
            line1_b = a[1] - a[0] * k
            line2_b = b[1] - b[0] * k
            line1 = LineString([(xmin, xmin * k + line1_b), (xmax, xmax * k + line1_b)])
            line2 = LineString([(xmin, xmin * k + line2_b), (xmax, xmax * k + line2_b)])
        else:  # If the y-coordinates of points a and b are the same
            line1 = LineString([(a[0], ymin), (a[0], ymax)])
            line2 = LineString([(b[0], ymin), (b[0], ymax)])

        # Create a LineString object from the contour
        cd = LineString(contour)
        
        # Find the intersection points of the contour with line1 and line2
        inter1 = cd.intersection(line1)
        inter2 = cd.intersection(line2)

        # Check if intersections are empty
        if inter1.is_empty or inter2.is_empty:
            return [(0, 0), (0, 0), 0, 0]
        
        # Handle different types of geometries for inter1
        if inter1.geom_type == 'MultiLineString':
            point1 = Point([inter1.geoms[0].xy[0][0], inter1.geoms[0].xy[1][0]])
        elif inter1.geom_type == 'Point':
            point1 = inter1
        elif inter1.geom_type == 'MultiPoint':
            point1 = inter1.geoms[0]
        elif inter1.geom_type == "GeometryCollection" and inter1.geoms:
            if inter1.geoms[0].geom_type == 'MultiPoint':
                point1 = inter1.geoms[0][0]
            elif inter1.geoms[0].geom_type == "Point":
                point1 = inter1.geoms[0]
            elif inter1.geoms[0].geom_type == 'LineString':
                point1 = Point([inter1.geoms[0].xy[0][0], inter1.geoms[0].xy[1][0]])
        
        # Handle different types of geometries for inter2
        if inter2.geom_type == 'MultiLineString':
            point2 = Point([inter2.geoms[0].xy[0][0], inter2.geoms[0].xy[1][0]])
        elif inter2.geom_type == 'Point':
            point2 = inter2
        elif inter2.geom_type == 'MultiPoint':
            point2 = inter2.geoms[0]
        elif inter2.geom_type == "GeometryCollection" and inter2.geoms:
            if inter2.geoms[0].geom_type == 'MultiPoint':
                point2 = inter2.geoms[0][0]
            elif inter2.geoms[0].geom_type == "Point":
                point2 = inter2.geoms[0]
            elif inter2.geoms[0].geom_type == 'LineString':
                point2 = Point([inter2.geoms[0].xy[0][0], inter2.geoms[0].xy[1][0]])

        # Calculate the distances from a to inter1 and b to inter2
        a_inter1_length = Point(a).distance(point1)
        b_inter2_length = Point(b).distance(point2)

        # Calculate the line length between point a and point b
        line_length = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        # Calculate the absolute value of the difference between a_inter1_length and b_inter2_length
        inter_difference = abs(a_inter1_length - b_inter2_length)
        # Calculate the angle (in radians)
        angle = math.atan(inter_difference / line_length)
        # Convert the angle to degrees
        angle = math.degrees(angle)

        # Calculate bounding box
        dis1=point1.distance(Point(a))
        dis2=point2.disjoint(Point(b))
        box_minx = int(min(point1.x, point2.x))
        box_miny = int(min(point1.y, point2.y))
        box_maxx = int(max(point1.x, point2.x))
        box_maxy = int(max(point1.y, point2.y))

        return [(box_minx, box_miny), (box_maxx, box_maxy), ((dis1+dis2)/2), angle]

    def getYL(lines,mode="aveGrid"):
        averageY=0
        maxY=0
        averageL=0
        maxL=0
        for l in lines:
            deltaY=np.abs(l[0][1]-l[1][1])
            averageY +=deltaY
            if deltaY>maxY:
                maxY=deltaY
            length=l[2]
            averageL+=length
            if length>maxL:
                maxL=length
        averageL = averageL / len(lines)
        averageY = averageY / len(lines)
        if mode == "aveGrid":
            wGrid = averageL
            hGrid = averageY
        if mode == "maxGrid":
            wGrid = maxL
            hGrid = maxY
        print("the width is : "+str(wGrid)+"; the height is : "+str(hGrid))
        return wGrid,hGrid
    ###plot grid on top of lines and display the color based on the lines

    def plotGridsOver(image_with_lines,lines,widthGrid,heightGrid, font, fontScale,
                         lineType,color='length'):
        #print(color)
        #grid = 10
        c_map = matplotlib.cm.get_cmap('bwr', 256)
        rgba_data = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(np.arange(0, 1.0, 1.0 / 256.0), bytes=True)
        rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))
        features_length = {}
        features_angle = {}
        #widthGrid = image_with_lines.shape[0] / grid
        #heightGrid = image_with_lines.shape[0] / grid
        gridoverlay=image_with_lines.copy()
        image_with_lines_tmp=image_with_lines.copy()
        grid1=int(image_with_lines.shape[0]/widthGrid)
        grid2=int(image_with_lines.shape[0]/heightGrid)
        for l in lines:
            for g1 in range(grid1):
                for g2 in range(grid2):
                    xmin = g1 * widthGrid
                    xmax = (g1 + 1) * widthGrid
                    ymin = g2 * heightGrid
                    ymax = (g2 + 1) * heightGrid
                    Inside = False
                    # if SubFun.checkPointInRect(l[0],[xmin,xmax,ymin,ymax]) \
                    #        or SubFun.checkPointInRect(l[1],[xmin,xmax,ymin,ymax]):
                    x, y = l[0]
                    if x > xmin and x < xmax and y > ymin and y < ymax:
                        Inside = True
                    x, y = l[1]
                    if x > xmin and x < xmax and y > ymin and y < ymax:
                        Inside = True
                    if Inside == False:
                        continue
                    # if l[3]>135:
                    #    l[3]=l[3]-180
                    #print(g1,g2)
                    try:
                        features_length[g1, g2].append(l[2])
                        features_angle[g1, g2].append(l[3])
                    except KeyError:
                        features_length[g1, g2] = list()
                        features_angle[g1, g2] = list()
                        features_length[g1, g2].append(l[2])
                        features_angle[g1, g2].append(l[3])
        avel_list=list()
        #vara_list=list()
        for g1, g2 in features_length:
            avel_list.append(int(np.mean(features_length[g1, g2])))
            #vara_list.append(int(np.std(features_angle[g1, g2])))

        #print(avel_list)
        for g1, g2 in features_length:
            ave_l = int(np.mean(features_length[g1, g2]))
            var_l = int(np.std(features_length[g1, g2]))
            smallangle = np.sum(np.array(features_angle[g1, g2]) < 45) / len(features_angle[g1, g2])
            largeangle = np.sum(np.array(features_angle[g1, g2]) >135) / len(features_angle[g1, g2])
            ### if small angle and large angle are majority,
            if smallangle >= 0.5 and len(features_length[g1, g2])>1:
                #print(features_angle[g1,g2])
                for angle_tmp in range(len(features_angle[g1, g2])):
                    if features_angle[g1, g2][angle_tmp] > 135:
                        features_angle[g1, g2][angle_tmp] = features_angle[g1, g2][angle_tmp] - 180
            if largeangle >= 0.5 and len(features_length[g1, g2])>1:
                #print(features_angle[g1,g2])
                for angle_tmp in range(len(features_angle[g1, g2])):
                    if features_angle[g1, g2][angle_tmp] < 45:
                        features_angle[g1, g2][angle_tmp] = features_angle[g1, g2][angle_tmp] +180
            ave_a = int(np.mean(features_angle[g1, g2]))
            var_a = int(np.std(features_angle[g1, g2]))
            #print(g1, g2,len(features_angle[g1,g2]), ave_l, var_l, ave_a, var_a)
            xmin = int(g1 * widthGrid)
            xmax = int((g1 + 1) * widthGrid)
            ymin = int(g2 * heightGrid)
            ymax = int((g2 + 1) * heightGrid)
            centerx = (xmin + xmax) / 2
            centery = (ymin + ymax) / 2
            # cv2.putText(unmerged, ",".join([str(len(features_angle[g1,g2]))+"  ",str(avel) , " "+str(varl)+'  ',str(avea),"  "+str(vara)]), (int(centerx),int(centery)), font, fontScale,
            #            fontColor, lineType)
            # cv2.rectangle(unmerged,(xmin,ymin),(xmax,ymax),fontColor)
            if color=="length":
                if np.max(avel_list)==np.min(avel_list):
                    density=255
                else:
                    density=int(255*(ave_l-np.min(avel_list))/(np.max(avel_list)-np.min(avel_list)))
                #fontColor=(0, density, 0)
            if color=="angle":
                density=int(255*var_a/90)
                #fontColor=(0, density, 0)
                #print(var_a,density)
                #cv2.putText(image_with_lines_tmp,
                #    str(ave_l) + "/" + str(var_l) +" "+str(ave_a) + "/" + str(var_a),
                #            (int(centerx), int(centery)), font, fontScale, (255,0,0), lineType)
            fontColor=(int(rgba_data[density][0][0]),int(rgba_data[density][0][1]),int(rgba_data[density][0][2]))
            #print(g1,g2,gridoverlay.shape,fontColor)
            cv2.putText(image_with_lines_tmp,
                        str(ave_l)  +","+ str(var_a),
                        (int(centerx), int(centery)), font, fontScale, (255,0,0), lineType)
            cv2.rectangle(gridoverlay, (xmin+1, ymin+1), (xmax-1, ymax-1), fontColor,-1)
        alpha=0.5
        image_with_lines_tmp=cv2.addWeighted(gridoverlay, alpha, image_with_lines_tmp, 1 - alpha,0, image_with_lines_tmp)
        return image_with_lines_tmp

    ##Plot lines on the image
    def printLines(image,lines, fontColor=(0, 0, 0), lineType=1, colorCode='length'):
        lengths=np.zeros(len(lines))
        for l in range(len(lines)):
            lengths[l]=lines[l][2]
        for l in lines:
            if colorCode == 'length':
                if lengths.max()>lengths.min():
                    density = int(255 /(lengths.max()-lengths.min())* (l[2]-lengths.min()))
                else:
                    density = 255
            else:
                density=255
            ###draw filtered and unmerged lines on global frame
            #cv2.line(outline, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])),
            #         (0, 0, density), 1)
            cv2.line(image, (int(l[0][0]), int(l[0][1])), (int(l[1][0]), int(l[1][1])),
                     (0,0,density), lineType)
        return image



    def detectLineFromContours(image,contours, font, fontScale,
                        fontColor, lineType):
        cvuint8 = np.copy(image)
        unfilted = np.copy(image)
        areas = list()
        lines = list()
        filteredLines = list()
        ### Iterate from all connected components
        for labelindex in range(len(contours)):
            # for labelindex in range(15,16):
            #cnt = np.nonzero(labels == labelindex)
            cnt=np.array(contours[labelindex])
            pnt=np.copy(cnt)
            pnt[:,1]=cnt[:,0]
            pnt[:,0]=cnt[:,1]

            rect = cv2.minAreaRect(cnt)
            w = rect[1][0]
            h = rect[1][1]
            angle = rect[2]
            if w > h:
                angle = int(90 - angle)
            else:
                angle = -int(angle)
            ###filter smaller lines
            if w * h < 10:  # and w*h>0:
                continue
            # if w*h >2000:
            #    continue
            areas.append(w * h)
            # print(w,h,w*h)
            length_rect = max([w, h])
            box = cv2.cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            if w > h:
                startpoint_ori = ((box[0] + box[1]) / 2)
                endpoint_ori = ((box[2] + box[3]) / 2)
            else:
                startpoint_ori = ((box[0] + box[3]) / 2)
                endpoint_ori = ((box[1] + box[2]) / 2)
            startpoint=np.copy(startpoint_ori)
            endpoint=np.copy(endpoint_ori)
            startpoint[0]=startpoint_ori[1]
            startpoint[1]=startpoint_ori[0]
            endpoint[0]=endpoint_ori[1]
            endpoint[1]=endpoint_ori[0]
            ###save lines (startpoint, endpoint, line length and line angle)
            lines.append([startpoint, endpoint, int(length_rect), int(angle)])
            cv2.line(unfilted, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])),
                     (0, 255, 0), 1)
            #cv2.drawContours(cvuint8, [box], 0, (0, 0, 255), 1)
            if w <= 2 * h and w >= h:
                continue
            if h < 2 * w and h >= w:
                continue
            if w * h < 100:  # and w*h>0:
                continue
            if w * h > 2000:
                continue
            ###draw filtered and unmerged lines on individual frame
            filteredLines.append([startpoint, endpoint, int(length_rect), int(angle)])
            cv2.line(cvuint8, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])),
                     RED, 1)

            cv2.putText(cvuint8, str(int(length_rect)) + "," + str(int(angle)), (box[0][1], box[0][0]), font, fontScale,
                        fontColor, lineType)
            linelength = cv2.norm(startpoint - endpoint)

            # cv2.putText(cvuint8, str(int(length_rect)) + "," + str(int(angle)), (box[0][0], box[0][1]), font, fontScale, fontColor, lineType)
            # [intX, intY, intW, intH]=cv2.boundingRect((labels == i).astype('uint8'))
            # if intW*intH >100:
            #    cv2.rectangle(cvuint8, (intX, intY), (intX+intW, intY+intH), (0, 255, 0), 1)
        #lines=np.array(lines)
        return cvuint8,lines,filteredLines,unfilted

    def fromRaw2binary(img,saveName,minimum_brightness=0.4,bglb=128,bgub=220,saveTofile=True):
        # cv2.imwrite("Neural Network Tryout/raw/raw_"+str(i)+".png",a[i,:,:])
        #img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)[:, :, 0]
        brightness = np.sum(image) / (255 * image.shape[0] * image.shape[1])
        ratio = brightness / minimum_brightness
        cvuint8 = cv2.convertScaleAbs(image, alpha=1 / ratio, beta=0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(cvuint8)

        ret1, th1 = cv2.threshold(cl1, bglb, 255, cv2.THRESH_TOZERO)
        ret1, th1 = cv2.threshold(th1, bgub, 255, cv2.THRESH_TOZERO_INV)
        th1 = th1.astype('uint8')
        #cv2.imwrite(saveName+"_clahe_thres128BG.png", th1)
        gb = cv2.GaussianBlur(th1, (1, 1), 0)
        #cv2.imwrite(saveName+"_clahe_thres128BG_gaussianBlur.png", gb)
        ret2, th2 = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if saveTofile:
            cv2.imwrite(saveName+"_clahe.png", cl1)
            cv2.imwrite(saveName+"_clahe_thres128BG_gaussianBlur_binaOTSU.png", th2)
        img_gray = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
        cl1_gray = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
        return img,cl1,img_gray,cl1_gray,th2


    ###Line detection from connected components
    def detectLineFromCCs(image,labels, font, fontScale,
                        fontColor, lineType,
                          ratioUB=10, ratioLB=3,
                          areaUB=2000, areaLB=80,
                          angleM=None,angleStd=20,
                          pixelLB=75,pixelUB=150,
                          lineWidthUB=3,printText=False):
        cvuint8 = cv2.convertScaleAbs(image)
        unfilted = cv2.convertScaleAbs(image)
        areas = list()
        lines = list()
        filteredLines = list()
        lines_detailed = list()
        filteredLines_detailed = list()

        ### Iterate from all connected components
        for labelindex in range(1, labels.max()+1):
            # for labelindex in range(15,16):
            cnt = np.nonzero(labels == labelindex)
            if np.sum(labels == labelindex) < pixelLB & np.sum(labels == labelindex) > pixelUB :
                continue
            pnts = np.zeros((len(cnt[0]), 2), dtype='float32')
            for j in range(len(cnt[0])):
                pnts[j, 0] = cnt[1][j]
                pnts[j, 1] = cnt[0][j]
            rect = cv2.minAreaRect(pnts)
            w = rect[1][0]
            h = rect[1][1]
            angle = rect[2]
            if w < h:
                angle = int(90 - angle)
            else:
                angle = -int(angle)
            ###filter smaller lines
            #if w * h < 10:  # and w*h>0:
            #    continue
            # if w*h >2000:
            #    continue
            areas.append(w * h)
            # print(w,h,w*h)
            length_rect = max([w, h])
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int64(box) #CHANGED BY ME
            if w > h:
                startpoint = ((box[0] + box[1]) / 2)
                endpoint = ((box[2] + box[3]) / 2)
            else:
                startpoint = ((box[0] + box[3]) / 2)
                endpoint = ((box[1] + box[2]) / 2)
            ###change row and cols

            ###save lines (startpoint, endpoint, line length and line angle)
            lines.append(
                [startpoint, endpoint, length_rect, angle])
            bwidth=min(w,h)
            bheight=max(w,h)
            if bwidth>0:
                ratio=bheight/bwidth
            else:
                ratio=0
            lines_detailed.append([startpoint[0],startpoint[1], endpoint[0],endpoint[1], length_rect, angle,bwidth,bheight, labelindex])
            cv2.line(unfilted, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])),
                     COLOR_GREEN, lineType)
            if printText:
                cv2.putText(unfilted, str(int(length_rect))+',' +str(int(bwidth)) + "," + str(int(angle)) + "," +  '%.2f' % ratio,
                        (box[0][0], box[0][1]), font, fontScale,
                        fontColor, lineType)
            cv2.drawContours(unfilted, [box], 0, COLOR_CYAN, 1)
            if bheight < ratioLB * bwidth:
                continue
            if bheight > ratioUB * bwidth:
                continue
            if w * h < areaLB:  # and w*h>0:
                continue
            if w * h > areaUB:
                continue
            if bwidth > lineWidthUB:
                continue
            if angleM is not None: #set = to None
                if (angle> angleM+angleStd) or (angle< angleM-angleStd) :   #use get angle std
                    continue
            ###draw filtered and unmerged lines on individual frame
            filteredLines.append(
                [startpoint, endpoint,int(length_rect), int(angle)])
            filteredLines_detailed.append(
                [startpoint[0], startpoint[1], endpoint[0], endpoint[1], length_rect, angle,bwidth,bheight,  labelindex])
            cv2.line(cvuint8, (int(startpoint[0]), int(startpoint[1])), (int(endpoint[0]), int(endpoint[1])),
                     COLOR_GREEN, lineType)

            if printText:
                cv2.putText(cvuint8, str(int(length_rect))+',' +str(int(bwidth))+ "," + str(int(angle))+"," + '%.2f' % ratio , (box[0][0], box[0][1]), font, fontScale,
                        fontColor, lineType)
            #linelength = cv2.norm(startpoint - endpoint)

            # cv2.putText(cvuint8, str(int(length_rect)) + "," + str(int(angle)), (box[0][0], box[0][1]), font, fontScale, fontColor, lineType)
            # [intX, intY, intW, intH]=cv2.boundingRect((labels == i).astype('uint8'))
            # if intW*intH >100:
            #    cv2.rectangle(cvuint8, (intX, intY), (intX+intW, intY+intH), (0, 255, 0), 1)
        #lines=np.array(lines)
        return cvuint8,lines,filteredLines,unfilted, np.array(lines_detailed), np.array(filteredLines_detailed)
    ## input
    ## lines : raw lines detected
    ## image : original image
    ## outline: outline of the ROI
    ## dist_thres (in pixels): merge lines only within this distance threshold
    ## pixel_density_thres (0 to 255): to connect the two lines, detemine valid pixel between them
    ## line_density_thres (0 to 1): to connect the two lines, detemine valid pixel ratio over the total distance between liens
    ## output
    ##
    def mergeLines(lines,image,outline,dist_thres=100,pixel_density_thres=60,line_density_thres=0.6):
        #print(lines)
        merged = list()
        mergedLines = list()
        lineInd = dict()
        curInd = 0
        cvuint8 = cv2.convertScaleAbs(image)
        for l1 in range(len(lines)):
            for l2 in range(l1+1,len(lines)):
                if l1 != l2:
                    p11 = lines[l1][0]
                    p12 = lines[l1][1]
                    p21 = lines[l2][0]
                    p22 = lines[l2][1]
                    dist = min(cv2.norm(p11 - p21), cv2.norm(p11 - p22), cv2.norm(p12 - p21), cv2.norm(p12 - p22))
                    nodepair = np.argmin(
                        (cv2.norm(p11 - p21), cv2.norm(p11 - p22), cv2.norm(p12 - p21), cv2.norm(p12 - p22)))
                    if dist > dist_thres:
                        continue
                    if nodepair == 0:
                        p1 = p11
                        p2 = p21
                        p3 = p12
                        p4 = p22
                    if nodepair == 1:
                        p1 = p11
                        p2 = p22
                        p3 = p12
                        p4 = p21
                    if nodepair == 2:
                        p1 = p12
                        p2 = p21
                        p3 = p11
                        p4 = p22
                    if nodepair == 3:
                        p1 = p12
                        p2 = p22
                        p3 = p11
                        p4 = p21
                    lineIter = SubFun.createLineIterator(p1.astype("int"), p2.astype("int"), cvuint8[:, :, 0])
                    ###check if the two lines are loosely connected in the original image
                    if (sum(lineIter[:, 2] > pixel_density_thres) / (lineIter.shape[0] + 1)) > line_density_thres:
                        merged.append([l1, l2])

                        if l1 in lineInd:
                            lineInd[l2] = lineInd[l1]
                        elif l2 in lineInd:
                            lineInd[l1] = lineInd[l2]
                        else:
                            # if l1 not in lineInd and l2 not in lineInd:
                            lineInd[l1] = curInd
                            lineInd[l2] = curInd
                            curInd += 1
                        linelength = cv2.norm(p3, p4)
                        density = int(255 / 200 * linelength)
                        ###print merged lines on individual frame
                        cv2.line(cvuint8, (int(p3[0]), int(p3[1])),
                                 (int(p4[0]), int(p4[1])), (0, 255,), 1)
                        ###print merged lines on all frames
                        #cv2.line(outline, (int(p3[0]), int(p3[1])),
                        #         (int(p4[0]), int(p4[1])), (0, density,), 1)
                        rect = cv2.minAreaRect(np.matrix((p3,p4),dtype='float32'))
                        w = rect[1][0]
                        h = rect[1][1]
                        angle = rect[2]
                        if w < h:
                            angle = int(90 - angle)
                        else:
                            angle = -int(angle)
                        mergedLines.append([p3,p4,linelength,angle])
        #merged = np.array(merged)
        #mergedLines = np.array(mergedLines)
        return merged,mergedLines, cvuint8
    def minDist(startpoint,endpoint,point):
        dnorm = np.linalg.norm(np.cross(endpoint - startpoint, startpoint - point)) / np.linalg.norm(
            endpoint - startpoint)
        dstart = np.linalg.norm(startpoint - point)
        dend = np.linalg.norm(endpoint - point)
        projection=np.dot(endpoint - startpoint, point-startpoint  )
        #print(projection,np.linalg.norm(endpoint-startpoint),dstart,dend,dnorm)
        if projection < 0 or projection > np.linalg.norm(endpoint-startpoint):
            return np.min([dstart,dend])
        else:
            return dnorm
    def findConvexHull(outline):
        ret, outline = cv2.threshold(outline, 50, 255, cv2.THRESH_BINARY)
        ol = outline[:, :, 0].copy()
        im2, contours, hierarchy = cv2.findContours(ol, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], True))
            for p in contours[i]:
                outline[p[0][1], p[0][0], 1] = 255
        drawing = np.zeros((outline.shape[0], outline.shape[1], 3), np.uint8)

        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0)  # green - color for contours
            color = (255, 0, 0)  # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)

    def createLineIterator(P1, P2, img):
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]
        # print(imageH,imageW,P1X,P1Y,P2X,P2Y)
        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)
        # predefine np array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

        return itbuffer


    def collectLinesOnBoundary(lines,edgepoint,distThres=300):
        collectLinesOnBoundary = dict()
        for p in edgepoint:
            mindist = distThres + 1
            for i in range(len(lines)):
                l = lines[i]
                dist = SubFun.point2lineDist(l[0], l[1], p)
                # print(dist,l,mindist)
                if dist < mindist:
                    minLine = i
                    mindist = dist
                    # print(i,mindist)
            if mindist <= distThres:
                collectLinesOnBoundary[p[0], p[1]] = minLine
        return collectLinesOnBoundary
    def point2lineDist(p1,p2,p):
        [x1,y1]=p1
        [x,y]=p
        [x2,y2]=p2
        area = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2)
        #print(x1,y1,x2,y2,x,y,area)
        dist = area / np.sqrt((x1 - x2) *(x1 - x2)  + (y1 - y2)* (y1 - y2))
        return dist

    def plotBoundary(dim1,dim2, lines,collectLinesOnBoundary,minv=-1,maxv=-1):
        #cv2.applyColorMap(,cv2.COLORMAP_JET,)
        c_map = matplotlib.cm.get_cmap('bwr', 256)
        rgba_data = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(np.arange(0, 1.0, 1.0 / 256.0), bytes=True)
        rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))

        #imageBoundry = image.copy()
        imageBoundry = np.zeros((dim1,dim2,3), np.uint8)
        if minv<0 or maxv<0:
            maxv=0
            minv=1000
            for x, y in collectLinesOnBoundary:
                l=lines[collectLinesOnBoundary[x,y]]
                #print(x,y,collectLinesOnBoundary[x,y],l,l[2])
                if maxv < l[2]:
                    maxv=l[2]
                if minv >l[2]:
                    minv=l[2]
        #print(maxv,minv)
        if maxv==minv:
            maxv=minv+1
        for x,y in collectLinesOnBoundary:
            #cv2.drawMarker(imageBoundry, [x, y], [255, 0, 0])
            l = lines[collectLinesOnBoundary[x, y]]
            norm_l=int((l[2]-minv)/(maxv-minv)*255)

            imageBoundry[x,y,:]=rgba_data[norm_l][0]
            #imageBoundry[x,y,1]=255
            #imageBoundry[x,y,2]=(l[2]-minv)/(maxv-minv)*255
        #im_color = cv2.applyColorMap(imageBoundry[:,:,0], cv2.COLORMAP_JET)
        return imageBoundry

    def tifContrast(image,offset=8,gamma=0.8):
            a1 = np.array(image)#.copy()
            a1= cv2.cvtColor(a1, cv2.COLOR_GRAY2BGR)
            cvuint8 = np.clip(a1 / 256, 0, 255)
            cvuint8 = np.clip(cvuint8 / cvuint8.min() * offset, 0, 255)
            cvuint8 = cvuint8.astype('uint8')  # *3
            gamma = 0.8
            lookUpTable = np.empty((1, 256), np.uint8)
            for pi in range(256):
                lookUpTable[0, pi] = np.clip(pow(pi / 255.0, gamma) * 255.0, 0, 255)
            res = cv2.LUT(cvuint8, lookUpTable)
            #cv2.imwrite(folder + subfolder + fname + str(i) + "_orig_gamma" + str(gamma) + ".png", res)
            #cv2.imwrite(folder + subfolder + fname + str(i) + "_orig.png", cvuint8)
            return cvuint8,res
    def GaussianCoef(x,y,d):
        dist=np.sqrt(x*x+y*y)
        return 1.0 - np.exp((-dist * dist) / (2 * d * d))
    def createGaussianFilter(width,height,cutoffrange,inv=False):
        img=np.ones((width,height))
        centerx=width/2
        centery=height/2
        for i in range(width):
            for j in range(height):
                img[i,j]= SubFun.GaussianCoef(i-centerx,j-centery,cutoffrange)
        if inv:
            img=1-img
        return img
    def backgroundsub(a):
        b=list()
        imagenumber=len(a)
        for i in range(imagenumber):
            a1, a1_gamma = SubFun.tifContrast(a[i],offset=10)
            print(a.shape)
            cv2.imwrite("image_ori"+str(i)+".png",a1)
            img, background = subtract_background_rolling_ball(a1[:,:,0], 10, light_background=False,
                                                               use_paraboloid=False,
                                                               do_presmooth=False)
            cv2.imwrite("image_subbackground"+str(i)+".png",img)
            filter_gauss_highpass = SubFun.createGaussianFilter(img.shape[0], img.shape[1], img.shape[0] / 16)
            filter_gauss_lowpass = SubFun.createGaussianFilter(img.shape[0], img.shape[1], img.shape[0] / 4, inv=True)
            highLowFilter = np.multiply(filter_gauss_lowpass, filter_gauss_highpass)

            f = np.fft.fft2(img.astype('float32'))
            fshift = np.fft.fftshift(f)

            # filter_resized = cv2.resize(filter, (2048,2048), interpolation = cv2.INTER_NEAREST)
            # cv2.imwrite("filterresize.png",filter_resized)
            fshift_filtered = fshift
            fshift_filtered[:, :, 0] = np.multiply(fshift[:, :, 0], highLowFilter)
            fshift_filtered[:, :, 1] = np.multiply(fshift[:, :, 1], highLowFilter)
            fshift_filtered[:, :, 2] = np.multiply(fshift[:, :, 2], highLowFilter)
            # fshift_filtered=fshift
            # fshift_filtered[1024]
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            #imgsub = 255 - img
            minpixel=img_back.min()
            img_back=(img_back-minpixel)*255/(255-minpixel)
            res, imgsubbi = cv2.threshold(img_back.astype('uint8'), 0, 255, cv2.THRESH_OTSU)
            cv2.imwrite("image_subbackground_fft_thres_bina"+str(i)+".png",imgsubbi)
            # img,back= subtract_background_rolling_ball(a1[:,:,0], 10, light_background=True, use_paraboloid=False, do_presmooth=True)
            #backsubimg = np.multiply(a1[:, :, 0], imgsubbi > 1)
            #cv2.imwrite("image_backsub"+str(i)+".png",backsubimg)
            #ret, backsubimg_bin = cv2.threshold(backsubimg, 0, 255, cv2.THRESH_OTSU)

            #backsubimg_bin, a1_gamma = SubFun.tifContrast(backsubimg,offset=10)
            #cv2.imwrite("image_backsub_bina" + str(i) + ".png", backsubimg_bin)
            b.append(img)
            # cv2.imwrite("image_orig" + str(i) + ".png", a1[:, :, 0] )
        return b