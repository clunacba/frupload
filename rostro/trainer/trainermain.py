# Import
from tkinter.messagebox import NO
import pygame
import cv2
import numpy as np
import cv2
import os
import pygame_textinput
import imutils
import encode_faces


button1 = None

# Initialize
pygame.init()


font = pygame.font.SysFont("Arial", 20)


class Button:
	"""Create a button, then blit the surface in the while loop"""

	def __init__(self, text,  pos, font, bg="black", feedback=""):
		self.x, self.y = pos
		self.font = pygame.font.SysFont("Arial", font)
		if feedback == "":
			self.feedback = "text"
		else:
			self.feedback = feedback
		self.change_text(text, bg)

	def change_text(self, text, bg="black"):
		"""Change the text whe you click"""
		self.text = self.font.render(text, 1, pygame.Color("White"))
		self.size = self.text.get_size()
		self.surface = pygame.Surface(self.size)
		self.surface.fill(bg)
		self.surface.blit(self.text, (0, 0))
		self.rect = pygame.Rect(self.x, self.y, self.size[0], self.size[1])
		
	def show(self):
		window.blit(button1.surface, (self.x, self.y))

	def click(self, event):
		x, y = pygame.mouse.get_pos()
		if event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				if self.rect.collidepoint(x, y):
					self.change_text(self.feedback, bg="red")
					encode_faces.encode()
					




# No arguments needed to get started
textinput = pygame_textinput.TextInputVisualizer()


# Create Window/Display
width, height = 1280, 720
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Webcam")


# Load Images
imgBackground = pygame.image.load("../graf/images/facialre1.png").convert_alpha()
imgBackground = pygame.transform.scale(imgBackground, (width, height))

marc = pygame.image.load("../graf/images/marc.png").convert_alpha()
marc = pygame.transform.scale(marc, (800, 600))

textbox = pygame.image.load("../graf/images/textbannerb.png").convert_alpha()
textbox = pygame.transform.scale(textbox, (400, 100))

textbox2 = pygame.image.load("../graf/images/textbannerw.png").convert_alpha()
textbox2 = pygame.transform.scale(textbox2, (300, 50))

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Load Images



# Rect
#rectNew = pygame.Rect(500, 200, 200, 200)
#Create Folder and path
def folder(name):
	if not os.path.exists(f'dataset/{name}'):
		print('Carpeta creada: dataset')
		os.makedirs(f'dataset/{name}')

	ruta = f'dataset/{name}/{name}_'
	return ruta

#Clasificador
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
k = None


# Webcam
cap = cv2.VideoCapture(1)
cap.set(3, 800)  # width
cap.set(4, 600)  # height

button1 = Button("Entrenar", (948, 350),font=30,bg="navy",feedback="Entrenado")
# Main loop
start = True
count = 0
while start:
	# Get Events
	events = pygame.event.get()
	# Feed it with events every frame
	textinput.update(events)
	#textinput_custom.update(events)

	

	# Modify attributes on the fly - the surface is only rerendered when .surface is accessed & if values changed
	#textinput_custom.font_color = [(c+10)%255 for c in textinput_custom.font_color]


	for event in events:
		if event.type == pygame.QUIT:
			start = False
			pygame.quit()
		button1.click(event)
		
		if event.type == pygame.KEYDOWN:
				
			if event.key == pygame.K_RETURN:
				if textinput.value:
					ruta = folder(textinput.value)
					#cv2.imshow('rostro',frame2)
					#cv2.waitKey(100)
					cv2.imwrite('{}_{}.jpg'.format(ruta,count),frame2)
					#cv2.imshow('rostro',frame)
					count = count +1
					print(f"User pressed enter! Input so far: {textinput.value}")
				else:
					print ('Nada')
	#button1.show()

	# Apply Logic
	
	# OpenCV
	#success, img = cap.read()
	#img = face_detect(img)
	###########################3
	ret,frame = cap.read()
	frame = cv2.flip(frame,1)
	"""gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()
	frame2 = frame.copy()

	faces = faceClassif.detectMultiScale(gray, 1.3, 5)

	k = cv2.waitKey(1)
	if k == 27:
		break

	for (x,y,w,h) in faces:
		p1 = x-50,y-50
		p2 = x+w+50,y+h+50
		cv2.rectangle(frame, (p1),(p2),(128,0,255),2)
		#rostro = auxFrame[y:y+h,x:x+w]
		rostro = auxFrame[y-50:y+h+50,x-50:x+w+50]
		rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
		#cv2.imshow('rostro',rostro)"""
		
	###########################
	img=frame
	frame2 = frame.copy()
	#cv2.flip(img,1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	imgRGB = np.rot90(imgRGB)
	frame = pygame.surfarray.make_surface(imgRGB).convert()
	frame = pygame.transform.flip(frame, False, False)
	
	window.blit(imgBackground, (0, 0))
	
	
	
	window.blit(frame, (100, 80))

	window.blit(marc, (20, 20))

	

	# Get its surface to blit onto the screen
	fonttitle1 = pygame.font.Font(None,40)
	title1 = fonttitle1.render("Insert Name Press ENTER", True, (255,0,0))
	
	window.blit(textbox,(800,80))
	window.blit(title1, (820, 137))
	window.blit(textinput.surface, (850, 100))
	
	window.blit(textbox2,(850,338))
	button1.show()

	# Update Display
	pygame.display.update()
	# Set FPS
	clock.tick(fps)
