#! /usr/bin/python
# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
from datetime import datetime
from graf.common import draw_rects, overlay_transparent
import json, base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt



"""def read(nombre):
	classes = []
	f = open('models/clients.json')
	data = json.load(f)
	print(data)
	for i in data['nombre']:
		classes.append(i)
	f.close()
	print (classes)"""
	#return classes


def sendalert(devid, name ,server,topico, img):
		client = mqtt.Client(client_id=devid)
		
		#client.username_pw_set(username="dsv_admin",password="Global*3522")
		try:
			client.connect(server, 1883, 60)
			msgjson=msg(devid,name, img)
			client.publish(topico,msgjson)
			print ("enviado")
			
		except:
			print ("NO conecto")

def msg(devid,name, img):
	t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	b64=base_64(img)
	d={"deviceid":devid,"cliente":name, "imagen":b64}
	return json.dumps(d)

def base_64(img):
	img = imutils.resize(img, 224)
	#print(type(img))
	_, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
	im_bytes = im_arr.tobytes()
	im_b64 = base64.b64encode(im_bytes)
	#print (im_b64.decode('UTF-8'))
	return im_b64.decode('UTF-8')








# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open('/home/cordiez/Cordiez/rostro/models/encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier('/home/cordiez/Cordiez/rostro/models/haarcascade_frontalface_default.xml')

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#rtsp = "rtsp://Cordiez:Cordiez123456@192.168.44.213:554/cam/realmonitor?channel=1&subtype=1"
rtsp = "rtsp://admin:qwer1234@192.168.0.101:554/cam/realmonitor?channel=1&subtype=1"
####comp = VideoStream(src=rtsp).start()
#comp = cv2.VideoCapture(0)
comp = cv2.VideoCapture(rtsp)
time.sleep(2.0)
name = None
nombre = None
band = True
env=True
mas100 = 0
#read()

# Main loop
start = True
while start:
	# Apply Logic

	# OpenCV
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	grab,frame = comp.read()
	if not grab:
		print ("No video")
		break
	###frame = comp.read()
	frame = imutils.resize(frame, width=450)
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	if boxes:
		pass
	else:
		
		band = False
		env=True
	
	

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		
		nombre = name
		draw_rects(frame, boxes,(0,0,255))
		y = top - 15 if top - 15 > 15 else top + 15
		frame
		#cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		
		
		dt = datetime.now()

		# getting the timestamp
		ts = datetime.timestamp(dt)
		print (name)
		actual1 = ts
		#print(actual1)
		if actual1 > mas100+6:
			if nombre != "Unknown":
				#sendalert("cord00000002", nombre, "192.168.100.17", "cordiez/ai/fr/001", frame)
				print ("Enviado 1")
				#band = True
				mas100=actual1
			

	#cv2.flip(img,1)
	imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	imgRGB = np.rot90(imgRGB)
	
	#print (name)	
	###cv2.imshow("aaa",frame)
	###if cv2.waitKey(1) & 0xFF == ord('q'):
        ###	break
	

	

	#if band == False:
	#	if env:
	#		sendalert("1122334455", nombre, "ng.drexgen.com", "central", frame)
	#		print ("bbbb")
	#		band = True
	#	print ("aaa")





