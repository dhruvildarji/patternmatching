import os
import time
import cv2
from vectors import *
import math
import struct
import numpy as np
import operator

PATH = "D:/LMU/Thesis2/Combine/pattern1/60targets/exp"
Num_Folders = 2
end = [(0,0),(33.8593602,-118.0779161),(33.8593435,-118.0779017)]       #pattern 1 end
#end = [(0,0), (33.8593478,-118.0779185),(33.859363,-118.0779019)]      #pattern2 end
#end = [(0,0), (33.8593645,-118.0779262),(33.8593553,-118.0779132)]      #East end
#end = [(0,0), (33.8593569,-118.0779331),(33.8593545,-118.0778996)]      #North end
C = 1.55
origin = (33.859106, -118.0781080)      #origin for Thesis 2
#origin = (33.9696400,-118.363686)      # origin for Thesis 1
GPS_ERROR = 0.4754
GPS_ERROR_MEAN = 0.5089
STD_GPS_ERROR = 0.4432
OVERLAP_FACTOR = 2
OVERLAP = 1
pic_size_pixels = (1058.0,600.0)
n = 86933.843345214
t = 86933.843345214
DDev_Line = 0.0185
SigmaDDev_Line = 0.0174
AlphaDev_Line = 3.8744
SigmaAlphaDev_Line = 3.2856
DDev_Tri = 0.0442
SigmaDDev_Tri = 0.0348
AlphaDev_Tri = 3.8427
SigmaAlphaDev_Tri = 3.1797
D_Tri = 1.0882
SigmaD_Tri = 1.0431
D_Line = 0.9974
SigmaD_Line = 1.1454
T_Tri =  D_Tri + C * SigmaD_Tri
T_Line =  D_Line + C * SigmaD_Line
T_Point = GPS_ERROR_MEAN + C * STD_GPS_ERROR
T_Same = (DDev_Line/2.0) + C * (SigmaDDev_Line/2.0)
T_Same = T_Same * 10
T_Line = T_Line * 1
T_Tri = T_Tri * 1


def Find_Perimeter(start_location, end_location, updated_locations, pic_size_length, pic_size_width):
	direction = np.subtract(start_location,end_location)
	mag = np.linalg.norm(direction)
	direction = direction / mag
	direction_L = np.multiply(direction,(pic_size_length/2.0))
	direction_B = (-direction[1], direction[0])
	direction_B = np.multiply(direction_B,(pic_size_width/2.0))	
	direction_R = (-direction_L[0], -direction_L[1])
	direction_T = (direction[1], -direction[0])
	direction_T = np.multiply(direction_T,(pic_size_width/2.0))
	perimeter_L = np.add(updated_locations,direction_L)
	perimeter_R = np.add(updated_locations,direction_R)
	perimeter_T = np.add(updated_locations,direction_T)
	perimeter_B = np.add(updated_locations,direction_B)
	return (perimeter_L,perimeter_R,perimeter_T,perimeter_B)

def Find_Contours(img):
	THRESHOLD_LOW = (0,90,100)#(0,90,100)
	THRESHOLD_HIGH = (10,255,255)
	THRESHOLD_LOW1 = (160,100,100)#160,100,100)
	THRESHOLD_HIGH1 = (179,255,255)
	MIN_RADIUS = 6
	img_filter = cv2.medianBlur(img,5)
	img_filter = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)
	img_binary1 = cv2.inRange(img_filter,THRESHOLD_LOW,THRESHOLD_HIGH,0)
	img_binary2 = cv2.inRange(img_filter,THRESHOLD_LOW1, THRESHOLD_HIGH1,0)
	img_binary = img_binary1 + img_binary2
	img_binary = cv2.dilate(img_binary, None, iterations = 7)	
	contours = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	return contours

def Find_Coordinates(contours,drone_coordinates,pic_size_pixels, meter_per_pixel, start_location, end_location):	
	direction = np.subtract(end_location,start_location)
	mag = np.linalg.norm(direction)
	direction = direction / mag
	angle = np.arctan2(direction[1], direction[0])
	list1 = {}
	list_xy = []
	if len(contours) > 0:	
		for i in range(len(contours)):	
			((x, y), radius) = cv2.minEnclosingCircle(contours[i])	
			x = int(x)
			y = int(y)
			find_pixels = (x-(pic_size_pixels[0]/2),(pic_size_pixels[1]/2)-y)
			pixels = Rotate_Vector(angle,find_pixels)
			find_meters = np.multiply(pixels,meter_per_pixel)
			coordinate = np.add(drone_coordinates,find_meters)
			list1[i] = coordinate
		return list1,list_xy
	else:
		return 0,0

def Move_Drone(start_location, end_location, step_size, num_image):
	direction = np.subtract(end_location,start_location)
	start_location_num = np.array([start_location[0],start_location[1]])
	end_location_num = np.array([end_location[0],end_location[1]])
	mag = np.linalg.norm(end_location_num - start_location_num)
	direction = direction / mag
	direction = np.multiply(direction,step_size)
	direction = np.multiply(direction,(num_image-1))
	center = start_location + direction
	return center

def Find_Corners(perimeters, centers):
	perimeter_L = perimeters[0]
	perimeter_R = perimeters[1]
	perimeter_T = perimeters[2]
	perimeter_B = perimeters[3]
	direction_L = np.subtract(perimeter_L,centers)	
	direction_R = np.subtract(perimeter_R,centers)	
	direction_T = np.subtract(perimeter_T,centers)	
	direction_B = np.subtract(perimeter_B,centers)	
	right_top_corner = np.add(direction_R,direction_T)
	left_top_corner = np.add(direction_L,direction_T)
	left_bottom_corner = np.add(direction_L,direction_B)
	right_bottom_corner = np.add(direction_R,direction_B)	
	r_t_corner = np.add(centers,right_top_corner)
	l_t_corner = np.add(centers,left_top_corner)
	l_b_corner = np.add(centers,left_bottom_corner)
	r_b_corner = np.add(centers,right_bottom_corner)
	return (r_t_corner,l_t_corner,l_b_corner,r_b_corner)

def point_inside_or_outside(point, center, coordinates):
	for i in range(len(coordinates)):
		vec = np.subtract(center, coordinates[i])
		point_vec = np.subtract(point, coordinates[i])
		a = np.dot(vec, point_vec)
		if a < 0 :
			return False
	return True

def if_overlapped(center1, perimeter1,image_corner1, center2, perimeter2, image_corner2):
	for i in range(4):
		corners_image2 = point_inside_or_outside(image_corner2[i], center1, perimeter1)
		corners_image1 = point_inside_or_outside(image_corner1[i], center2, perimeter2)
		perimeter_image2 = point_inside_or_outside(perimeter2[i], center1, perimeter1)
		perimeter_image1 = point_inside_or_outside(perimeter1[i], center2, perimeter2)
		if corners_image2 == True or corners_image1 == True or perimeter_image2 == True or perimeter_image1 == True:
			return True
	return False

def adjust_overlap(center1, perimeter1, Target_Coordinates1, center2, perimeter2, Target_Coordinates2,image_corner2, GPS_ERROR, OVERLAP_FACTOR):
	Targets = []
	image_corners = []
	Distance = GPS_ERROR * OVERLAP_FACTOR
	direction = np.subtract(center1, center2)
	mag = np.linalg.norm(direction)
	direction = np.divide(direction,mag)
	direction = np.multiply(direction, Distance)
	center2 = np.add(center2,direction)
	perimeter2 = np.add(perimeter2, direction)
	image_corners = np.add(image_corner2, direction)
	for i in range(len(Target_Coordinates2)):
		Targets.append(np.add(Target_Coordinates2[i], direction))
	return center2, perimeter2, Targets, image_corners

def Find_Overlapped_Targets(center1, perimeter1, Target_Coordinates1, center2, perimeter2, Target_Coordinates2):
	Image1_Overlapped = []
	Image2_Overlapped = []
	for i in range(len(Target_Coordinates1)):
		a = point_inside_or_outside(Target_Coordinates1[i], center1, perimeter1)
		b = point_inside_or_outside(Target_Coordinates1[i], center2, perimeter2) 
		if a and b is True:
			Image1_Overlapped.append(Target_Coordinates1[i])	
	for j in range(len(Target_Coordinates2)):
		a = point_inside_or_outside(Target_Coordinates2[j], center1, perimeter1)
	
		b = point_inside_or_outside(Target_Coordinates2[j], center2, perimeter2)
		if a and b is True:
			Image2_Overlapped.append(Target_Coordinates2[j])
	return Image1_Overlapped, Image2_Overlapped

def find_direction(start_location, end_location):
	direction = np.subtract(end_location,start_location)
	direction_mag = np.linalg.norm(direction)
	direction = direction / direction_mag
	return direction

def rotate_image(angle, center2, perimeter2, target_coordinates2, corner2):
	new_target_coordinates = []
	new_perimeter_coordinates = []
	new_corner_coordinates = []
	difference = math.radians(angle)
	for i in range(len(target_coordinates2)):
		vec = np.subtract(target_coordinates2[i], center2)
		vec = np.add(Rotate_Vector(difference,vec),center2)
		new_target_coordinates.append(vec)
	for j in range(len(perimeter2)):
		perimeter_vec = np.subtract(perimeter2[j], center2)
		perimeter_vec = np.add(Rotate_Vector(difference,perimeter_vec), center2)
		new_perimeter_coordinates.append(perimeter_vec)
		corner_vec = np.subtract(corner2[j], center2)
		corner_vec = np.add(Rotate_Vector(difference,corner_vec), center2)
		new_corner_coordinates.append(corner_vec)
	return center2, new_perimeter_coordinates, new_target_coordinates, new_corner_coordinates

def Order_Targets(Target_Coordinates1, center1, Target_Coordinates2, center2):
	Image1_Targets = []
	Image2_Targets = []	
	cmp_vec1 = {}
	cmp_vec2 = {}
	vect = np.subtract(center1, center2)
	vect1 = (-vect[1],vect[0])
        vect_mag = np.linalg.norm(vect)
	vect_unit12 = vect/vect_mag
        vect_unit21 = np.multiply(vect_unit12,-1)
	for i in range(len(Target_Coordinates1)):
		target_vec = np.subtract(Target_Coordinates1[i], center1)
		cmp_vec1[i] = np.dot(vect_unit12, target_vec)
	for j in range(len(Target_Coordinates2)):
		target_vec = np.subtract(Target_Coordinates2[j], center2)
		cmp_vec2[j] = np.dot(vect_unit21, target_vec)	
	c1 = [c[0] for c in sorted(enumerate(cmp_vec1.values()),key=lambda i:i[1])]
	b1 = [b[0] for b in sorted(enumerate(cmp_vec2.values()),key=lambda i:i[1])]
	for i in range(len(Target_Coordinates1)):
		Image1_Targets.append(Target_Coordinates1[c1[i]])		
	for j in range(len(Target_Coordinates2)):
		Image2_Targets.append(Target_Coordinates2[b1[j]])
	return Image1_Targets, Image2_Targets

def Find_Angle(vec1, vec2):
	cosang = np.dot(vec1, vec2)
	sinang = np.linalg.norm(np.cross(vec1,vec2))
	return math.degrees(np.arctan2(sinang, cosang))

def Find_Small_Angle(a,b,c):
	Angle1 = {}
	ang_vec_ab = np.subtract(b,a)	
	ang_vec_ac = np.subtract(c,a)
	ang_vec_ba = np.subtract(a,b)	
	ang_vec_bc = np.subtract(c,b)
	ang_vec_ca = np.subtract(a,c)	
	ang_vec_cb = np.subtract(b,c)
	Angle1[0] = Find_Angle(ang_vec_ab,ang_vec_ac)
	Angle1[1] = Find_Angle(ang_vec_ba,ang_vec_bc)
	Angle1[2] = Find_Angle(ang_vec_ca,ang_vec_cb)	
	sumation = Angle1[0] + Angle1[1] + Angle1[2]	
	Angle1_sorted = sorted(Angle1.items(), key = operator.itemgetter(1))
	Angle2_sorted_max = sorted(Angle1.items(), key = operator.itemgetter(1),reverse = True)
	if Angle2_sorted_max[0][0] == 0:
		Angle_vec11_max = ang_vec_ab
		Angle_vec12_max = ang_vec_ac			
	elif Angle2_sorted_max[0][0] == 1:
		Angle_vec11_max = ang_vec_ba
		Angle_vec12_max = ang_vec_bc			
	else:
		Angle_vec11_max = ang_vec_ca
		Angle_vec12_max = ang_vec_cb	
	if Angle1_sorted[0][0]	== 0:
		Angle_vec11 = ang_vec_ab
		Angle_vec12 = ang_vec_ac
	elif Angle1_sorted[0][0] == 1:
		Angle_vec11 = ang_vec_ba
		Angle_vec12 = ang_vec_bc
	else:
		Angle_vec11 = ang_vec_ca
		Angle_vec12 = ang_vec_cb		
	return Angle_vec11, Angle_vec12, Angle_vec11_max, Angle_vec12_max ,Angle1_sorted		

def swap_vertices(a,b,c):
	Angle1 = {}
	ang_vec_ab = np.subtract(b,a)	
	ang_vec_ac = np.subtract(c,a)
	ang_vec_ba = np.subtract(a,b)	
	ang_vec_bc = np.subtract(c,b)
	ang_vec_ca = np.subtract(a,c)	
	ang_vec_cb = np.subtract(b,c)
	Angle1[0] = Find_Angle(ang_vec_ab,ang_vec_ac)
	Angle1[1] = Find_Angle(ang_vec_ba,ang_vec_bc)
	Angle1[2] = Find_Angle(ang_vec_ca,ang_vec_cb)	
	sumation = Angle1[0] + Angle1[1] + Angle1[2]	
	Angle1_sorted = sorted(Angle1.items(), key = operator.itemgetter(1))
	if Angle1_sorted[1][0]	== 0:
		Angle_vec11 = ang_vec_ab
		Angle_vec12 = ang_vec_ac

	elif Angle1_sorted[1][0] == 1:
		Angle_vec11 = ang_vec_ba
		Angle_vec12 = ang_vec_bc
	else:
		Angle_vec11 = ang_vec_ca
		Angle_vec12 = ang_vec_cb		
	return Angle_vec11, Angle_vec12

def change_vertices(angles):
	temp0 = angles[0][1]
	temp1 = angles[1][1]
	temp2 = angles[2][1]
	tempr = temp0
	temp0 = temp1
	temp1 = tempr
	angles1 = ((angles[0][0],temp0),(angles[1][0],temp1),(angles[2][0],temp2))
	return angles1

def Find_Distance(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3):
	vec11 = np.subtract(I1_T1,I1_T2)
	vec12 = np.subtract(I1_T2,I1_T3)			
	vec13 = np.subtract(I1_T1,I1_T3)
	vec21 = np.subtract(I2_T1,I2_T2)
	vec22 = np.subtract(I2_T2,I2_T3)			
	vec23 = np.subtract(I2_T1,I2_T3)
	vec21_mag = np.linalg.norm(vec21)
	vec22_mag = np.linalg.norm(vec22)
	vec23_mag = np.linalg.norm(vec23)
	vec2x = (vec21_mag, vec22_mag, vec23_mag)
	vec2x = np.sort(vec2x)
	vec11_mag = np.linalg.norm(vec11)
	vec12_mag = np.linalg.norm(vec12)
	vec13_mag = np.linalg.norm(vec13)
	vec1x = (vec11_mag, vec12_mag, vec13_mag)
	vec1x = np.sort(vec1x)
	diff = np.linalg.norm(np.subtract(vec1x,vec2x))
	return diff

def Find_Unit_Vector(vec):
	mag = np.linalg.norm(vec)
	vec = np.divide(vec,mag)
	return vec

def Find_Vectors_Max_Angle(vec_s1,vec_s2,vec_l1,vec_l2):
	vs1 = Find_Unit_Vector(vec_s1)
	vs2 = Find_Unit_Vector(vec_s2)
	vl1 = Find_Unit_Vector(vec_l1)
	vl2 = Find_Unit_Vector(vec_l2)
	if Find_Angle(vec_s1,vec_l1) == 180:
		return vec_l1,vec_l2
	if Find_Angle(vec_s1,vec_l2) == 180:
		return vec_l2,vec_l1
	if Find_Angle(vec_s2,vec_l1) == 180:
		return vec_l1,vec_l2
	if Find_Angle(vec_s2,vec_l2) == 180:
		return vec_l2,vec_l1

def Rotate_Vector(rot,vec2): 
	angle = np.arctan2(vec2[1], vec2[0])
	new_angle = angle + rot
	r = np.linalg.norm(vec2)
	vec2 = (r*math.cos(new_angle), r*math.sin(new_angle))
	return vec2	

def Find_Rotation(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3):
	vec_s11,vec_s12,vec_l11,vec_l12,angles1 =  Find_Small_Angle(I1_T1, I1_T2, I1_T3)
	vec_s21,vec_s22,vec_l21,vec_l22,angles2 =  Find_Small_Angle(I2_T1, I2_T2, I2_T3)
	vec_small1, vec_large1 = Find_Vectors_Max_Angle(vec_s11,vec_s12,vec_l11,vec_l12)
	vec_small2, vec_large2 = Find_Vectors_Max_Angle(vec_s21,vec_s22,vec_l21,vec_l22)
	rot = Find_Angle(vec_small1, vec_small2)
	rot = math.radians(rot)
	perpendicular_min = (-vec_small2[1],vec_small2[0])
	if np.dot(perpendicular_min,vec_small1) < 0:
		rot = -rot
	vec_small2_check = Rotate_Vector(rot,vec_small2)
	vec_large2_check = Rotate_Vector(rot,vec_large2)
	if Find_Angle(vec_small1,vec_large1) + Find_Angle(vec_small2,vec_large2) > 270:
		if Find_Angle(vec_small1, vec_small2) > Find_Angle(vec_small1, vec_large2):
			vec_s21,vec_s22 = swap_vertices(I2_T1, I2_T2, I2_T3)				
	else:
		if np.dot(vec_large1,vec_large2_check) < 0:
			vec_s21,vec_s22 = swap_vertices(I2_T1, I2_T2, I2_T3)	
	vec_21_unit = Find_Unit_Vector(vec_s21)
	vec_22_unit = Find_Unit_Vector(vec_s22)
	vec_11_unit = Find_Unit_Vector(vec_s11)
	vec_12_unit = Find_Unit_Vector(vec_s12)
	Ref1 = np.add(vec_11_unit,vec_12_unit)
	Ref2 = np.add(vec_21_unit,vec_22_unit)
	angle = Find_Angle(Ref1,Ref2)
	if np.dot((-Ref2[1],Ref2[0]),Ref1) < 0 :
		angle = -angle
	return angle

def Find_Deviation(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3):
	distance = Find_Distance(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3)
	angle = Find_Rotation(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3)
	angle = abs(angle)
	centroid1 = np.add(I1_T1,I1_T2)
	centroid1 = np.add(centroid1,I1_T3)
	centroid1 = np.divide(centroid1,3)	 
	centroid2 = np.add(I2_T1,I2_T2)
	centroid2 = np.add(centroid2,I2_T3)
	centroid2 = np.divide(centroid2,3)
	location = np.linalg.norm(np.subtract(centroid1,centroid2))
	if location > T_Point:
                location = 1000
	deviation = deviation_formula_triangle(distance,angle,location)
	return deviation,angle

def Find_Deviation_Paths(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3):
	distance = Find_Distance(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3)
	angle = Find_Rotation(I1_T1, I1_T2, I1_T3, I2_T1, I2_T2, I2_T3)
	deviation = deviation_path_formula(distance,angle)
	return deviation

def Find_Deviation_Path_Lines(I1_T1, I1_T2, I2_T1, I2_T2):
	direction = (0,1)
	dist_1 = np.linalg.norm(np.subtract(I1_T1,I1_T2))
	dist_2 = np.linalg.norm(np.subtract(I2_T1,I2_T2))
	vec1 = Find_Unit_Vector(np.subtract(I1_T1, I1_T2))
	angle_1 = Find_Angle(direction,vec1)
	vec2 = Find_Unit_Vector(np.subtract(I2_T1, I2_T2))
	angle_2 = Find_Angle(direction,vec2)
	dist_diff = np.subtract(dist_2, dist_1)
	angle_diff = np.subtract(angle_2,angle_1)	
	deviation = deviation_path_formula(dist_diff,angle_diff)
	return deviation

def similar_triangles(Image1_Targets, Image2_Targets):
	p,q,r = Image1_Targets[0],Image1_Targets[1],Image1_Targets[2]
	difference = []
	angles = []
	for i in range(len(Image2_Targets)-2):
		a,b,c = Image2_Targets[i],Image2_Targets[i+1],Image2_Targets[i+2]
		deviation,angle = Find_Deviation(p,q,r,a,b,c)
		difference.append(deviation)
		angles.append(angle)		
	s = difference.index(min(difference))
	angle = angles[s]
	Image2_Triangle = (Image2_Targets[s],Image2_Targets[s+1],Image2_Targets[s+2])
	Image1_Triangle = (Image1_Targets[0],Image1_Targets[1],Image1_Targets[2])
	
	return Image1_Triangle, Image2_Triangle,s,angle, min(difference)

def Find_One_Target(Image1_Targets,Image2_Targets, GPS_ERROR):
	dist = []
	if len(Image1_Targets) == 1:
		Image1 = Image1_Targets
		Image2 = Image2_Targets
	else:
		Image1 = Image2_Targets
		Image2 = Image1_Targets
	for i in range(len(Image2)):
		dist.append(np.linalg.norm(np.subtract(Image1[0],Image2[i])))
	distance = min(dist)
	if distance < GPS_ERROR:
		list1 = Image1
		list2 = Image2[dist.index(min(dist))]
	else:
		list1 = []
		list2 = []
	return list1,list2

def One_Target(Image1_Targets, center1, Image2_Targets, center2):
	deviation = []
	for i in range(len(Image2_Targets)):
		dist = np.linalg.norm(np.subtract(Image1_Targets[0], Image2_Targets[i]))
		deviation.append(dist)
	index = deviation.index(min(deviation))
	Image1_Coordinate = Image1_Targets[0]
	Image2_Coordinate = Image2_Targets[index]		
	return Image1_Coordinate,Image2_Coordinate,index, min(deviation)		

def deviation_formula_triangle(distance,angle,location):
	deviation = math.sqrt((distance/(DDev_Tri+SigmaDDev_Tri))**2 + (angle/(AlphaDev_Tri + SigmaAlphaDev_Tri))**2 +(location/(GPS_ERROR+STD_GPS_ERROR))**2)
	return deviation

def deviation_formula_line(distance,angle,location):
	deviation = math.sqrt((distance/(DDev_Line+SigmaDDev_Line))**2 + (angle/(AlphaDev_Line + SigmaAlphaDev_Line))**2 +(location/(GPS_ERROR+STD_GPS_ERROR))**2)
	return deviation

def Find_Deviation_Lines(I1_T1, I1_T2, I2_T1, I2_T2):
        centroid1 = np.add(I1_T1,I1_T2)
	centroid1 = np.divide(centroid1,2)
	centroid2 = np.add(I2_T1,I2_T2)
	centroid2 = np.divide(centroid2,2)
	location = np.linalg.norm(np.subtract(centroid1,centroid2))
	if location > T_Point:
                location = 1000
	direction = (0,1)
	vec1 = Find_Unit_Vector(np.subtract(I1_T1, I1_T2))
	angle_1 = Find_Angle(direction,vec1)
	if angle_1 > 90:
                angle_1 = 180 - angle_1
        if (vec1[0] < 0 and vec1[1] > 0) or (vec1[0] > 0 and vec1[1] < 0):
                angle_1 = angle_1 * -1
	vec2 = Find_Unit_Vector(np.subtract(I2_T1, I2_T2))
	angle_2 = Find_Angle(direction,vec2)
        if angle_2 > 90:
                angle_2 = 180 - angle_2
        if (vec2[0] < 0 and vec2[1] > 0) or (vec2[0] > 0 and vec2[1] < 0):
                angle_2 = angle_2 * -1
        angle_diff = abs(np.subtract(angle_2,angle_1))
        dist_1 = np.linalg.norm(np.subtract(I1_T1,I1_T2))       
	dist_2 = np.linalg.norm(np.subtract(I2_T1,I2_T2))
	dist_diff = abs(np.subtract(dist_2, dist_1))
	dev = deviation_formula_line(dist_diff,angle_diff, location)
	return dev,np.subtract(angle_2,angle_1)

def Two_Targets(Image1_Targets, center1, Image2_Targets, center2, perimeter2):
	p,q = Image1_Targets[0],Image1_Targets[1]
	difference = []
	angles = []
	for i in range(len(Image2_Targets)-1):
		a,b = Image2_Targets[i],Image2_Targets[i+1]
		deviation,angle = Find_Deviation_Lines(p,q,a,b)
		difference.append(deviation)
		angles.append(angle)	
	s = difference.index(min(difference))
	angle = angles[s]
	Image2_Triangle = (Image2_Targets[s],Image2_Targets[s+1])
	Image1_Triangle = (Image1_Targets[0],Image1_Targets[1])
	return Image1_Triangle, Image2_Triangle,s,angle, min(difference)
	
def Find_Two_Target(Image1_Targets, center1, Image2_Targets, center2, perimeter2):
	deviation = []
	angles = []
	distances = []
	flag = False
	if len(Image1_Targets) == 2:
		flag = False
		Image1 = Image1_Targets
		Image2 = Image2_Targets
	else:
		flag = True
		Image1 = Image2_Targets
		Image2 = Image1_Targets
	direction = (0,1)
	dist_1 = np.linalg.norm(np.subtract(Image1[0], Image1[1]))
	vec1 = Find_Unit_Vector(np.subtract(Image1[0], Image1[1]))
	angle_1 = Find_Angle(direction,vec1)
	for i in range(len(Image2)-1):
		dev = Find_Deviation_Lines(Image1[0], Image1[1],Image2[i+1], Image2[i])
		deviation.append(dev)
	if min(deviation) < T_Line:
                index = deviation.index(min(deviation))
                if flag == True:
                        list2 = Image1	
                        list1 = [Image2[index], Image2[index+1]]
                else:
                        list1 = Image1	
                        list2 = [Image2[index], Image2[index+1]]
                return list1, list2
        else:
                return [],[]

def Find_Paths_Two_Targets(center1, Image1_Targets, center2, Image2_Targets, Image2_Targets_actual, angle):
	direction12 = Find_Unit_Vector(np.subtract(center2,center1))
	vec = {}
	Image2 = []
	for i in range(len(Image2_Targets)):
		aux = np.subtract(Image2_Targets[i],center2)
		vec[i] = np.dot(aux,direction12)
	d1 = [d[0] for d in sorted(enumerate(vec),key=lambda i:i[1])]
	for i in range(len(Image2_Targets)):
		Image2.append(Image2_Targets[d1[i]])
	vec = {}
	Image1_aux = []
	for i in range(len(Image1_Targets)):
		Image1_aux.append(np.array(Image1_Targets[i]))
	for i in range(len(Image1_aux)):
		aux = np.subtract(Image1_aux[i],center1)
		vec[i] = np.dot(aux,direction12)
	d1 = [d[0] for d in sorted(enumerate(vec),key=lambda i:i[1],reverse = True)]
	Image1 = []
	for i in range(len(Image1_aux)):
		Image1.append(Image1_aux[d1[i]])
	Lines1 = []
	Lines2 = []	
	for i in range(len(Image1) - 1):
		Lines1.append((Image1[i],Image1[i+1]))
	for i in range(len(Image2) - 1):
		Lines2.append((Image2[i],Image2[i+1]))
	len1 = len(Lines1)
	len2 = len(Lines2)
	X = (np.array((-1,-1)),np.array((-1,-1)))
	for i in range(len2-1):
		Lines1.insert(0,X)
	for i in range(len1-1):
		Lines2.append(X)
	len1 = len(Lines1)
	deviation = []
	count_list = []
	for i in range(len1):
		count = 0
		dev = 0
		for j in range(len(Lines1)):
			dist1 = np.linalg.norm(np.subtract(Lines1[j][0],X[0]))
			dist2 = np.linalg.norm(np.subtract(Lines2[j][0],X[0]))
			if int(dist1) == 0 or int(dist2) == 0:
				pass
			else:
				count = count + 1
				aux = Find_Deviation_Path_Lines(Lines1[j][0],Lines1[j][1],Lines2[j][0],Lines2[j][1])
				dev = aux + dev
		count_list.append(count)
		dev = dev/count
		deviation.append(dev)
		del(Lines1[0])
		del(Lines2[-1])
	index = deviation.index(min(deviation))	
	list3 = []
	list1 = []
	list2 = []
	angle = -angle
	angle = math.radians(angle)
	for i in range(count_list[index]+1):
		vec = np.subtract(Image2[i],center2)
		vec = np.add(Rotate_Vector(angle,vec),center2)
		list3.append(vec)
	for j in range(len(list3)):
		dist1 = []
		for k in range(len(Image2)):
			dist = np.subtract(list3[j],Image2_Targets_actual[k])
			dist1.append(np.linalg.norm(dist))
		list2.append(Image2_Targets_actual[dist1.index(min(dist1))])
		list1.append((Image1_aux[j]))
	return list1, list2

def Find_Paths(Image1_Targets ,Image2_Targets):
	list1 = []
	list2 = []
	Image2_Targets = list(Image2_Targets)
	for i in range(len(Image1_Targets)):
		dist = []
		for j in range(len(Image2_Targets)):
			dist.append(np.linalg.norm(np.subtract(Image1_Targets[i], Image2_Targets[j])))
		if min(dist) <  T_Same:
			list1.append(Image1_Targets[i])
			list2.append(Image2_Targets[dist.index(min(dist))])
			if len(Image2_Targets) > 1:
				del(Image2_Targets[dist.index(min(dist))])
			else:
				break
	return list1, list2

def Find_Centroid(coordinates):	
	temp = 0
	for i in range(len(coordinates)):
		temp = np.add(temp,coordinates[i])
	temp = temp / len(coordinates)
	return temp

def Find_Zoom_Factor(Image1_Triangle,Image2_Triangle, centroid1, centroid2, center2):
	centroidz = np.add(center2,np.subtract(centroid1,centroid2))
	centroidz = centroid1
	z = []
	T1 = []
	T2 = []
	Targets1 = []
	Targets2 = []	
	for i in range(len(Image1_Triangle)):
		dist = []
		for j in range(len(Image2_Triangle)):
			dist.append(np.linalg.norm(np.subtract(Image1_Triangle[i],Image2_Triangle[j])))
			index = dist.index(min(dist))
			coordinate = Image2_Triangle[index]
		Targets1.append(Image1_Triangle[i])
		Targets2.append(coordinate)
		T1.append(np.subtract(Image1_Triangle[i],centroidz))
		T2.append(np.subtract(coordinate,centroidz))

	v11, v12, v13 = T1[0], T1[1], T1[2]	
	v21, v22, v23 = T2[0], T2[1], T2[2]			
	z_fact = np.add(np.dot(v11,v21),np.dot(v12,v22))	
	z_fact = np.add(z_fact,np.dot(v13,v23))
	z_fact = np.divide(z_fact, (np.linalg.norm(v21))**2 + (np.linalg.norm(v22))**2 + (np.linalg.norm(v23))**2)
	return z_fact

def Find_Zoom_Factor_Lines(Image1_Lines,Image2_Lines, centroid1, centroid2, center2):
	centroidz = np.add(center2,np.subtract(centroid1,centroid2))
	centroidz = centroid1
	z = []
	T1 = []
	T2 = []
	Targets1 = []
	Targets2 = []
	dist1 = np.linalg.norm(np.subtract(Image1_Lines[0],Image2_Lines[0]))
	dist2 = np.linalg.norm(np.subtract(Image1_Lines[0],Image2_Lines[1]))
	if dist1 < dist2:
                v11 = np.subtract(Image1_Lines[0],centroidz)
                v12 = np.subtract(Image1_Lines[1],centroidz)
                v21 = np.subtract(Image2_Lines[0],centroidz)
                v22 = np.subtract(Image2_Lines[1],centroidz)
        else:
                v11 = np.subtract(Image1_Lines[0],centroidz)
                v12 = np.subtract(Image1_Lines[1],centroidz)
                v21 = np.subtract(Image2_Lines[1],centroidz)
                v22 = np.subtract(Image2_Lines[0],centroidz)
	z_fact = np.add(np.dot(v11,v21),np.dot(v12,v22))	
	z_fact = np.divide(z_fact, (np.linalg.norm(v21))**2 + (np.linalg.norm(v22))**2)
	return z_fact

def Apply_Zoom(target2, Z, centroid1, centroid2, center2):
	centroidz = np.add(center2,np.subtract(centroid1,centroid2))
	centroidz = centroid1
	target2 = np.add(np.multiply(Z,target2),np.multiply((1-Z),centroidz))
	return target2	

def Apply_De_Zoom(list2, Z, centroid1, centroid2, center2):
	centroidz = np.add(center2,np.subtract(centroid1,centroid2))
	centroidz = centroid1
	target = np.multiply(centroidz,(Z-1))
	target = np.add(list2,target)
	target = np.divide(target, Z)
	return target
	
def Multiple_Drones_Counting_Targets(first_image,second_image,center1, perimeter1, Target_Coordinate1, image_corner1, center2, perimeter2, Target_Coordinate2, image_corner2,pixel_targets1,pixel_targets2, img_num1, img_num2):
	a = []
	b = []
	shifted_center2, shifted_perimeter2, shifted_Target_Coordinates2, shifted_image_corner2 = adjust_overlap(center1, perimeter1, Target_Coordinate1, center2, perimeter2, Target_Coordinate2, image_corner2, GPS_ERROR, OVERLAP_FACTOR)
	with open(PATH+str(1)+"maximum_overlap "+str(first_image)+str(second_image), 'w') as f:
                f.write('Targets_x = [')		
                for ii in range(len(shifted_Target_Coordinates2)):
                        f.write(str(shifted_Target_Coordinates2[ii][0]))	
                        f.write(',')
                f.write(']')
                f.write(';')
                f.write('\n')			
                f.write('Targets_y = [')
                f.write('\n')			
                for ii in range(len(shifted_Target_Coordinates2)):
                        f.write(str(shifted_Target_Coordinates2[ii][1]))	
                        f.write(',')
                f.write(']')
                f.write(';')
                f.write('\n')
                f.write('center_x = [')		
                f.write(str(shifted_center2[0]))
                f.write(']')
                f.write(';')
                f.write('\n')
                f.write('center_y = [')
                f.write(str(shifted_center2[1]))
                f.write(']')
                f.write(';')
                f.write('\n')
                f.write('image_corners_x = [')
                for ii in range(len(shifted_image_corner2)):
                        f.write(str(shifted_image_corner2[ii][0]))	
                        f.write(',')
                f.write(str(shifted_image_corner2[0][0]))        
                f.write(']')
                f.write(';')
                f.write('\n')
                f.write('image_corners_y = [')
                for ii in range(len(shifted_image_corner2)):
                        f.write(str(shifted_image_corner2[ii][1]))	
                        f.write(',')
                f.write(str(shifted_image_corner2[0][1]))       
                f.write(']')
                f.write(';')
                f.write('\n')
                f.close()
	T_F = if_overlapped(center1, perimeter1, image_corner1, shifted_center2, shifted_perimeter2, shifted_image_corner2)
	if T_F is True:
		Image1_Overlapped, shifted_Image2_Overlapped = Find_Overlapped_Targets(center1, perimeter1, Target_Coordinate1, shifted_center2, shifted_perimeter2, shifted_Target_Coordinates2)
		Image1_Overlapped, shifted_Image2_Overlapped = Order_Targets(Image1_Overlapped, center1, shifted_Image2_Overlapped, shifted_center2)
		Image1_Overlapped, Image2_Overlapped = Order_Targets(Image1_Overlapped, center1, Target_Coordinate2, center2)
        	Image2_Overlapped = Image2_Overlapped[:len(shifted_Image2_Overlapped)]
                pixel_targets1, pixel_targets2 = Order_Targets(pixel_targets1, center1, pixel_targets2, shifted_center2)
                pixel_targets1 = pixel_targets1[:len(Image1_Overlapped)]
                pixel_targets2 = pixel_targets2[:len(Image2_Overlapped)]
                Image1_Targets = Image1_Overlapped
                Image2_Targets = Image2_Overlapped
		with open(PATH+str(1)+"/text_one"+str(first_image)+str(second_image), 'w') as f:
			f.write('\n')
			f.write('Targets1_x = [')		
			for ii in range(len(Image1_Targets)):
				f.write(str(Image1_Targets[ii][0]))	
				f.write(',')
                        f.write(']')
			f.write('\n')			
			f.write('Targets1_y = [')		
			for ii in range(len(Image1_Targets)):
				f.write(str(Image1_Targets[ii][1]))	
				f.write(',')
                        f.write(']')
			f.write('\n')			
			f.write('Targets2_x = [')		
			for ii in range(len(Image2_Targets)):
				f.write(str(Image2_Targets[ii][0]))	
				f.write(',')
                        f.write(']')
			f.write('\n')			
			f.write('Targets2_y = [')
			f.write('\n')			
			for ii in range(len(Image2_Targets)):
				f.write(str(Image2_Targets[ii][1]))	
				f.write(',')
                        f.write(']')
			f.write('\n')
			f.write('center1_x = [')		
			f.write(str(center1[0]))
                        f.write(',')
                        f.write(str(center2[0]))
                        f.write(']')
                        f.write('\n')
			f.write('center1_y = [')		
			f.write(str(center1[1]))
                        f.write(',')
                        f.write(str(center2[1]))
                        f.write(']')
                        f.write('\n')
			f.write('image_corners1_x = [')
                        for ii in range(len(image_corner1)):
				f.write(str(image_corner1[ii][0]))	
				f.write(',')
			f.write(']')
			f.write('\n')
			f.write('image_corners1_y = [')
		        for ii in range(len(image_corner1)):
				f.write(str(image_corner1[ii][1]))	
				f.write(',')
			f.write(']')
                        f.write('\n')
			f.write('image_corners2_x = [')
                        for ii in range(len(image_corner2)):
				f.write(str(image_corner2[ii][0]))	
				f.write(',')
			f.write(']')
			f.write('\n')
			f.write('image_corners2_y = [')
		        for ii in range(len(image_corner2)):
				f.write(str(image_corner2[ii][1]))	
				f.write(',')
                        f.write(']')
			f.close()
		if len(Image1_Targets) == 0 or len(Image2_Targets) == 0:
			return a,b
		if len(Image1_Targets) < 3 or len(Image2_Targets) < 3:
			if len(Image1_Targets) == 1 or len(Image2_Targets) == 1:
				list1, list2 = Find_Paths(Image1_Targets, Image2_Targets)
				return list1,list2
			elif len(Image1_Targets) == 2 or len(Image2_Targets) == 2:
				Image1_Line, Image2_Line,s,angle,deviation = Two_Targets(Image1_Targets, center1, Image2_Targets, center2, perimeter2)
				centroid1 = Find_Centroid(Image1_Line)
                                centroid2 = Find_Centroid(Image2_Line)
                                aux,perimeter2,target2, corner2 = rotate_image(angle, centroid2, perimeter2, Image2_Targets, image_corner2)
                                aux3,aux2,Image2_Line, aux1 = rotate_image(angle, centroid2, perimeter2, Image2_Line, image_corner2)
                                direction = np.subtract(centroid1,centroid2)
                                dist = np.linalg.norm(direction)
                                aux, perimeter2, target2,image_corner2 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, target2, corner2, dist, 1)
                                aux, aux2, Image2_Line,aux1 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, Image2_Line, corner2, dist, 1)
                                center2_aux = np.add(center2, direction)
                                Z = Find_Zoom_Factor_Lines(Image1_Line,Image2_Line, centroid1, centroid2, center2)                                
                                if Z > 1:
                                        Z = Z * 1                                        
                                else:
                                        Z = Z * 1
                                target2 = Apply_Zoom(target2, Z, centroid1, centroid2, center2)
				with open(PATH+str(1)+"/zoomed"+str(first_image)+str(second_image), 'w') as f:
                                                f.write('Targets2_x = [')		
                                                for ii in range(len(Image2_Line)):
                                                        f.write(str(Image2_Line[ii][0]))	
                                                        f.write(',')
                                                f.write(']')
                                                f.write('\n')			
                                                f.write('Targets2_y = [')
                                                f.write('\n')			
                                                for ii in range(len(Image2_Line)):
                                                        f.write(str(Image2_Line[ii][1]))	
                                                        f.write(',')
                                                f.write(']')
                                                f.write('\n')
                                                f.write('center2_x = [')		
                                                f.write(str(center2_aux[0]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('center2_y = [')		
                                                f.write(str(center2_aux[1]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('image_corners2_x = [')
                                                for ii in range(len(image_corner2)):
                                                        f.write(str(image_corner2[ii][0]))	
                                                        f.write(',')
                                                f.write(str(image_corner2[0][0]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('image_corners2_y = [')
                                                for ii in range(len(image_corner2)):
                                                        f.write(str(image_corner2[ii][1]))	
                                                        f.write(',')
                                                f.write(str(image_corner2[0][1]))
                                                f.write(']')
                                                f.close()
                                list1, list2 = Find_Paths(Image1_Targets, target2)
                                list2 = Apply_De_Zoom(list2, Z, centroid1, centroid2, center2)
                                aux, perimeter2, list2,image_corner2 = adjust_overlap(centroid2, perimeter1, Image1_Targets, centroid1, perimeter2, list2, image_corner2, dist, 1)
                                center2,perimeter2,list2, corner2 = rotate_image(-angle, centroid2, perimeter2, list2, image_corner2)
				a = list1
				b = list2
				print "Image has 2 Targets"
			else:
				print "Images has no taregts"
				a,b = [],[]
			return a,b
		else:
			Image1_Targets_aux = []
			for i in Image1_Targets:
				Image1_Targets_aux.append(i)
			Image1_Triangle, Image2_Triangle,s,angle,deviation = similar_triangles(Image1_Targets_aux, Image2_Targets)
                        while(deviation > T_Tri):	
				del(Image1_Targets_aux[0])						
				if len(Image1_Targets_aux) == 2:
					print "Went in to 2"
					Image1_Targets_aux = []
					for i in Image1_Targets:
						Image1_Targets_aux.append(i)	
					Image1_Line, Image2_Line,s,angle,deviation = Two_Targets(Image1_Targets, center1, Image2_Targets, center2, perimeter2)
					while(deviation > T_Line):
						del(Image1_Targets_aux[0])
						if len(Image1_Targets_aux) == 1:
                                                        print "went in to 1s"
                                                        list1, list2 = Find_Paths(Image1_Targets, Image2_Targets)
                                                        return list1,list2
						Image1_Line, Image2_Line,s,angle,deviation = Two_Targets(Image1_Targets_aux, center1, Image2_Targets, center2, perimeter2)	
					centroid1 = Find_Centroid(Image1_Line)
					centroid2 = Find_Centroid(Image2_Line)
					center2,perimeter2,target2, corner2 = rotate_image(angle, centroid2, perimeter2, Image2_Targets, image_corner2)
					aux3,aux2,Image2_Line, aux1 = rotate_image(angle, centroid2, perimeter2, Image2_Line, image_corner2)
					direction = np.subtract(centroid1,centroid2)
					dist = np.linalg.norm(direction)
					aux, perimeter2, target2,image_corner2 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, target2, corner2, dist, 1)
					aux, aux2, Image2_Line,aux1 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, Image2_Line, corner2, dist, 1)
					center2_aux = np.add(center2, direction)
					Z = Find_Zoom_Factor_Lines(Image1_Line,Image2_Line, centroid1, centroid2, center2)
                                        if Z > 1:
                                                print "Zoom above 1"
                                                Z = Z * 1                                                
                                        else:
                                                print "Zoom below 1"
                                                Z = Z * 1
                                        target2 = Apply_Zoom(target2, Z, centroid1, centroid2, center2)
                                        with open(PATH+str(1)+"/zoomed"+str(first_image)+str(second_image), 'w') as f:
                                                f.write('Targets2_x = [')		
                                                for ii in range(len(target2)):
                                                        f.write(str(target2[ii][0]))	
                                                        f.write(',')
                                                f.write(']')
                                                f.write('\n')			
                                                f.write('Targets2_y = [')
                                                f.write('\n')			
                                                for ii in range(len(target2)):
                                                        f.write(str(target2[ii][1]))	
                                                        f.write(',')
                                                f.write(']')
                                                f.write('\n')
                                                f.write('center2_x = [')		
                                                f.write(str(center2_aux[0]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('center2_y = [')		
                                                f.write(str(center2_aux[1]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('image_corners2_x = [')
                                                for ii in range(len(image_corner2)):
                                                        f.write(str(image_corner2[ii][0]))	
                                                        f.write(',')
                                                f.write(str(image_corner2[0][0]))
                                                f.write(']')
                                                f.write('\n')
                                                f.write('image_corners2_y = [')
                                                for ii in range(len(image_corner2)):
                                                        f.write(str(image_corner2[ii][1]))	
                                                        f.write(',')
                                                f.write(str(image_corner2[0][1]))
                                                f.write(']')
                                                f.close()                                        
					list1, list2 = Find_Paths(Image1_Targets, target2)
					list2 = Apply_De_Zoom(list2, Z, centroid1, centroid2, center2)
					aux, perimeter2, list2,image_corner2 = adjust_overlap(centroid2, perimeter1, Image1_Targets, centroid1, perimeter2, list2, image_corner2, dist, 1)
					center2,perimeter2,list2, corner2 = rotate_image(-angle, centroid2, perimeter2, list2, image_corner2)
					a = list1
					b = list2
					return a,b
				Image1_Triangle, Image2_Triangle,s,angle,deviation = similar_triangles(Image1_Targets_aux, Image2_Targets)
			centroid1 = Find_Centroid(Image1_Triangle)
			centroid2 = Find_Centroid(Image2_Triangle)
			direction = np.subtract(centroid1,centroid2)
			center2_aux = np.add(center2,direction)
			dist = np.linalg.norm(direction)
			center2,perimeter2,target2, corner2 = rotate_image(angle, centroid2, perimeter2, Image2_Targets, image_corner2)
			aux3,aux2,Image2_Triangle, aux1 = rotate_image(angle, centroid2, perimeter2, Image2_Triangle, image_corner2)
			aux, perimeter2, target2,image_corner2 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, target2, corner2, dist, 1)
			aux, aux2, Image2_Triangle,aux1 = adjust_overlap(centroid1, perimeter1, Image1_Targets, centroid2, perimeter2, Image2_Triangle, corner2, dist, 1)
			Z = Find_Zoom_Factor(Image1_Triangle,Image2_Triangle, centroid1, centroid2, center2)
			if Z > 1:
        			Z = Z * 1        			
        		else:
                                Z = Z * 1
			target2 = Apply_Zoom(target2, Z, centroid1, centroid2, center2)
			with open(PATH+str(1)+"/zoomed"+str(first_image)+str(second_image), 'w') as f:
                                f.write('Targets2_x = [')		
                                for ii in range(len(target2)):
                                        f.write(str(target2[ii][0]))	
                                        f.write(',')
                                f.write(']')
                                f.write('\n')			
                                f.write('Targets2_y = [')
                                f.write('\n')			
                                for ii in range(len(target2)):
                                        f.write(str(target2[ii][1]))	
                                        f.write(',')
                                f.write(']')
                                f.write('\n')
                                f.write('center2_x = [')	
                                f.write(str(center2_aux[0]))
                                f.write(']')
                                f.write('\n')
                                f.write('center2_y = [')		
                                f.write(str(center2_aux[1]))
                                f.write(']')
                                f.write('\n')
                                f.write('image_corners2_x = [')
                                for ii in range(len(image_corner2)):
                                        f.write(str(image_corner2[ii][0]))	
                                        f.write(',')
                                f.write(str(image_corner2[0][0]))
                                f.write(']')
                                f.write('\n')
                                f.write('image_corners2_y = [')
                                for ii in range(len(image_corner2)):
                                        f.write(str(image_corner2[ii][1]))	
                                        f.write(',')
                                f.write(str(image_corner2[0][1]))
                                f.write(']')
                                f.close()
			list1, list2 = Find_Paths(Image1_Targets, target2)
			list2 = Apply_De_Zoom(list2, Z, centroid1, centroid2, center2)
			aux, perimeter2, list2,image_corner2 = adjust_overlap(centroid2, perimeter1, Image1_Targets, centroid1, perimeter2, list2, image_corner2, dist, 1)
			center2,perimeter2,list2, corner2 = rotate_image(-angle, centroid2, perimeter2, list2, image_corner2)
			a = list1
			b = list2
			return a,b			
	return a,b

def Find_Matched_Coordinate(coordinate,list_of_coordinates):
	dist = []
	for i in range(len(list_of_coordinates)):
		dist.append(np.linalg.norm(np.subtract(coordinate,list_of_coordinates[i][0])))
	index = dist.index(min(dist))
	coordinate = list_of_coordinates[index]
	return coordinate[0], index

def putext(img,targets_m,targets_p,no1,corner):
        a = 0
        w = []
        new_pixels = []
        for i in targets_m:
                target = np.subtract(i,corner)
                target = np.multiply(target,(1058/pic_size_length,-600/pic_size_width))
                new_pixels.append((int(target[0]),int(target[1])))
        for i in new_pixels:
                targets_m0 = round(targets_m[a][0],5)
                targets_m1 = round(targets_m[a][1],5)
                he = "("+str(targets_m0)+","+str(targets_m1)+")"
                w = img
                q = cv2.putText(w,he,i,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 2)
                cv2.imshow('img',q)
                cv2.waitKey(0)
                a = a + 1
        cv2.imwrite('newimage'+str(no1)+'.bmp',w)

updated_locations = {}
centers = {}
Coordinates = {}
Target_Coordinates = {}
perimeter_points = {}
perimeter = {}
corners = {}
image_corners = {}
num_images = {}
total_num_of_images = 0
latitude = []
location = {}
yaw = {}
yaw_vec = {}
img_num = {}
pixel_targets = {}
k = 0
kk=0
for i in range(1,Num_Folders+1):
	file_path = PATH + str(i) +"/file.txt"
	f = open(file_path,"r") 
	lines = f.readlines()
	lines_covered = 0
	lines[0] = lines[0].replace("\n","")
	Num_Images = lines[0]
	num_images[i] = int(Num_Images)
	lines[1] = lines[1].replace("\n","")
	step_size = lines[1]
	STEP_SIZE = int(step_size)
	lines[2] = lines[2].replace("\n","")
	vfov = lines[2]
	vfov = float(vfov)
	lines_covered = 3
	for im in range(num_images[i]):
		lines[im+lines_covered] = lines[im+lines_covered].replace("\n","")
		lat,lon = lines[im+lines_covered].split(',')
		lat = list(lat)
		lat = "".join(lat)
		latitude.append(float(lat))
		if im == 0:		 
			location[im+1] = ((float(lon)-origin[1])*n,(float(lat)-origin[0])*t)
		else:
                        location[im+1] = ((float(lon)-origin[1])*n,(float(lat)-origin[0])*t)
	lines_covered = 3 + num_images[i]
	for im in range(num_images[i]):
		yaw_value = lines[im+lines_covered].replace("\n","")
		yaw[im+1] = float(yaw_value)
		yaw_vec[im+1] = (math.cos(math.radians(float(yaw_value))),-math.sin(math.radians(float(yaw_value))))
	start_location = location[1] 
	location_end = ((float(end[i][1]) - origin[1])*n,(float(end[i][0]) - origin[0])*t)
	end_location = location_end
	pic_size_length = STEP_SIZE + OVERLAP + GPS_ERROR
	meter_per_pixel = pic_size_length / (pic_size_pixels[0]*1.0) 
	pic_size_width = (pic_size_pixels[1]*pic_size_length)/pic_size_pixels[0]	
	for j in range(1,num_images[i]+1):
                kk = kk + 1
		img_path = PATH + str(i) + '/Image' + str(j) + '.jpg'
		img = cv2.imread(img_path)
                contour = Find_Contours(img)
                updated_locations[j] = location[j]
		Coordinates[j],pixeltargets = Find_Coordinates(contour,updated_locations[j], pic_size_pixels, meter_per_pixel, start_location, end_location)	
		perimeter_points[j] = Find_Perimeter(start_location, end_location, updated_locations[j], pic_size_length, pic_size_width)
		corners[j] = Find_Corners(perimeter_points[j], updated_locations[j])
		ang1 = Find_Angle(np.subtract(corners[j][3],corners[j][2]),(1,0))
		img_num[kk] = img
		pixel_targets[kk] = pixeltargets
		angle = 180 - yaw[j]
		if contour != []:
                        updated_locations[j],perimeter_points[j],Coordinates[j], corners[j] = rotate_image(angle,updated_locations[j],perimeter_points[j],Coordinates[j],corners[j])
	centers[i] = updated_locations
	Target_Coordinates[i] = Coordinates
	perimeter[i] = perimeter_points
	image_corners[i] = corners
	updated_locations = {}
	Coordinates = {}
	perimeter_points = {}
	corners = {}
k = 0
centers_new = {}
Target_Coordinates_new = {}
perimeter_new = {}
image_corners_new = {}
for i in range(1, Num_Folders+1):
	total_num_of_images = total_num_of_images + num_images[i]
	for j in range(1, num_images[i]+1):
		k = k + 1
		image_corners_new[k] = image_corners[i][j]
		centers_new[k] = centers[i][j]
		Target_Coordinates_new[k] = Target_Coordinates[i][j]
                perimeter_new[k] = perimeter[i][j]
                if not Target_Coordinates_new[k]:
                        break
                with open(PATH+str(1)+"/Image no "+str(k), 'w') as f:
                        f.write('Targets_x = [')		
                        for ii in range(len(Target_Coordinates_new[k])):
                                f.write(str(Target_Coordinates_new[k][ii][0]))	
                                f.write(',')
                        f.write(']')
                        f.write(';')
                        f.write('\n')			
                        f.write('Targets_y = [')
                        f.write('\n')			
                        for ii in range(len(Target_Coordinates_new[k])):
                                f.write(str(Target_Coordinates_new[k][ii][1]))	
                                f.write(',')
                        f.write(']')
                        f.write(';')
                        f.write('\n')
                        f.write('center_x = [')		
                        f.write(str(centers_new[k][0]))
                        f.write(']')
                        f.write(';')
                        f.write('\n')
                        f.write('center_y = [')
                        f.write(str(centers_new[k][1]))
                        f.write(']')
                        f.write(';')
                        f.write('\n')
                        f.write('image_corners_x = [')
                        for ii in range(len(image_corners_new[k])):
                                f.write(str(image_corners_new[k][ii][0]))	
                                f.write(',')
                        f.write(str(image_corners_new[k][0][0]))        
                        f.write(']')
                        f.write(';')
                        f.write('\n')
                        f.write('image_corners_y = [')
                        for ii in range(len(image_corners_new[k])):
                                f.write(str(image_corners_new[k][ii][1]))	
                                f.write(',')
                        f.write(str(image_corners_new[k][0][1]))       
                        f.write(']')
                        f.write(';')
                        f.write('\n')
                        f.close()
images = {}		
for i in range(1,k+1):
	images[i] = {}
        if not Target_Coordinates_new[i]:
                pass
        else:                
                for j in range(len(Target_Coordinates_new[i])):
                        
                        images[i][j] = {}
                        images[i][j] = [Target_Coordinates_new[i][j]]
for i in range(1, total_num_of_images):	
	for k in range(i+1, total_num_of_images+1):
		list1, list2 = Multiple_Drones_Counting_Targets(i,k,centers_new[i], perimeter_new[i], Target_Coordinates_new[i], image_corners_new[i] , centers_new[k], perimeter_new[k], Target_Coordinates_new[k], image_corners_new[k],pixel_targets[i],pixel_targets[k],img_num[i],img_num[k])
		for p in range(len(list1)):
			coordinate1, index1 = Find_Matched_Coordinate(list1[p],images[i])
			coordinate2, index2 = Find_Matched_Coordinate(list2[p],images[k])
			images[i][index1].append(coordinate2)
			images[k][index2].append(coordinate1)		
Total = 0
yu = 0
y1 = 0
for i in range(1, total_num_of_images+1):
        if not Target_Coordinates_new[i]:
                pass
        else:                        
                for k in range(len(Target_Coordinates_new[i])):
                        if images[i][k] == []:
                                pass
                        else:
                                if float(len(images[i][k])) == 1:
                                        yu = yu + 1
                                elif float(len(images[i][k])) == 3:
                                        y1 = y1 + 1
                                else:
                                        pass                                                
                                Total = Total + (1/float(len(images[i][k])))
print "Total number of Targets",Total
