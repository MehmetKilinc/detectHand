import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

el_nesnesi = mp.solutions.hands
eller = el_nesnesi.Hands(static_image_mode = False)
#static_image_mode : false:detect true:track default:false
#max_number_hands = 1 :max el sayısı default:1
cizgiler = mp.solutions.drawing_utils

while True:
	ret, frame = cap.read()
	#ret kamera açılıp açılmadığını belirtir.True False
	resimRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
	#resmi bgr den rgb ye çevirme
	sonuc = eller.process(resimRGB)
	#print(sonuc.multi_hand_landmarks)
	#eli görmeyince none,görünce koordinat döndürür.	
	if sonuc.multi_hand_landmarks:
		for i in sonuc.multi_hand_landmarks:
			cizgiler.draw_landmarks(frame , i , el_nesnesi.HAND_CONNECTIONS)
			#çizgileri göster

			for id,j in enumerate(i.landmark):
				#j : koordinatlar  #j.x j.y j.z : koordinatları alma
				#id : eklem numarası
				h , w , c = frame.shape
				cx , cy = int(j.x * w) , int(j.y * h)
				if id == 4:
					cv2.circle(frame , (cx , cy) , 9 , (255,0,0) , cv2.FILLED)
	cv2.imshow("video",frame)
	if cv2.waitKey(1) & 0XFF == ord("q"):
		break
cv2.destroyAllWindows()
