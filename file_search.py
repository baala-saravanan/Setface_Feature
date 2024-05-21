import os
import vlc
from time import sleep
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
              
#               d='/home/pi/Hearsight/English/Face_Recognition/resources/audio/'+name
              d='/home/rock/Desktop/Hearsight/English/set_face_who/resources/audio/'+name
#              print("hai")
              media = vlc.MediaPlayer(d)
              media.play()
              sleep(3)
              media.stop()
              media.release()
              
              
#               media = vlc.MediaPlayer("/home/pi/Hearsight/English/English/thank_you.mp3")         
#               os.system (c)
              
              
        
#print(find("person1.wav",'/home/pi/.Hear_Sight/Face_Recognition/resources/audio'))