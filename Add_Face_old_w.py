import os
import time
import wget
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
import pickle5
import cv2
import face_recognition
import numpy as np
from PIL import Image
from config import *
from English.set_face_who import file_search
import sounddevice as sd
import soundfile as sf
import vlc
import time
import shutil
import gpio as GPIO
import subprocess
import random
import sys
from pydub import AudioSegment

sys.path.insert(0, '/home/rock/Desktop/Hearsight/')
from play_audio import GTTSA

play_machine_voice = GTTSA()



if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

GPIO.setup(450, GPIO.IN)
GPIO.setup(421, GPIO.IN)
GPIO.setup(447, GPIO.IN)
GPIO.setup(448, GPIO.IN)

GPIO_TRIGECHO = 501

GPIO.setup(GPIO_TRIGECHO,GPIO.OUT)  # Initial state as output
GPIO.output(GPIO_TRIGECHO, False)
    
    
def get_audio_duration(filename):
    audio = AudioSegment.from_file(filename)
    duration_in_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    return duration_in_seconds

def play_rec(voice):
    print(voice)
    filename = voice
    media = vlc.MediaPlayer(filename)
    media.play()
    duration = get_audio_duration(filename)
    time.sleep(float(duration) + 0.5)  # Adjust sleep time for playback speed
    media.stop()
    media.release()
    
def get_name():
    base_name = "person"
    existing_files = os.listdir(AUDIODIR)
    existing_files = [file for file in existing_files if os.path.isfile(os.path.join(AUDIODIR, file))]
    existing_numbers = []
    for file_name in existing_files:
        if file_name.startswith(base_name):
            try:
                number = int(file_name[len(base_name):].split('.')[0])  # Extract number before any extension
                existing_numbers.append(number)
            except ValueError:
                pass
    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 0
    new_name = f"{base_name}{next_number}"
    return new_name

def voice_rec(name):
    fs = 48000
    duration = 3
    try:
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
    except Exception as e:
        pass
    
    audio_path = os.path.join(AUDIODIR, f"{name}.wav")

    if not os.path.exists(audio_path):
        sf.write(audio_path, myrecording, fs)
        time.sleep(1)
        
    print("Audio saved to : ",audio_path)
    return audio_path

def delete_face(name_to_delete):
    
    base_folders = [RAW_IMG_DIR, EMBEDDINGS_DIR, AUDIODIR]

    for base_folder in base_folders:
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if name_to_delete in file:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

            for dir_name in dirs:
                if name_to_delete in dir_name:
                    folder_path = os.path.join(root, dir_name)
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")


class Recognizer:
    def __init__(self):
        
        if not os.path.isdir(RAW_IMG_DIR):
            os.makedirs(RAW_IMG_DIR)

        if not os.path.isdir(EMBEDDINGS_DIR):
            os.makedirs(EMBEDDINGS_DIR)
            
        if not os.path.isdir(AUDIODIR):
            os.makedirs(AUDIODIR)
        
        
        self.persons = os.listdir(RAW_IMG_DIR)
        self.detector = face_recognition.face_locations
        
        self.persons = os.listdir(RAW_IMG_DIR)
        
        self.detector = face_recognition.face_locations
        
        self.feature_extractor = face_recognition.face_encodings
        
        self.normalizer = Normalizer(norm='l2')
        
        self.label_enc = LabelEncoder()
        
        self.model = SVC(kernel='linear', probability=True)
        
        self.VERBOSE = VERBOSE
        
        tf.compat.v1.enable_eager_execution()


        if os.path.isfile(RECOGNIZER_PATH):
            
            self.recognizer = pickle5.load(open(RECOGNIZER_PATH, 'rb'))
            
            with open(MAPPING_PATH, "rb") as f:
                unpickler = pickle5.Unpickler(f)
                self.label_dict = unpickler.load()


    def augument(self, pixels):
        
        images = [pixels]
        seed = random.randint(1, 2)
        
        image = tf.image.random_brightness(pixels, max_delta=0.4, seed=seed)
        images.append(image.numpy())
        
        image = tf.image.random_contrast(pixels, lower=0.3, upper=0.8, seed=seed)
        images.append(image.numpy())

        image = tf.image.random_jpeg_quality(pixels, min_jpeg_quality=40, max_jpeg_quality=60, seed=seed)
        images.append(image.numpy())

        image = tf.image.random_saturation(pixels, lower=0.4, upper=0.6, seed=seed)
        images.append(image.numpy())

        return images

    def collect_images(self, path, name):
        play_machine_voice.play_machine_audio("press confirm button 3 times to capture image.mp3")
        cap = cv2.VideoCapture(1)
        i = 1
        
        while i <= NO_IMG_PER_PERSON:
            ret, frame = cap.read()
            key = cv2.waitKey(1)
            if GPIO.input(447)== True:
                
                if self.check_face(frame):
                    cv2.imwrite(f"{path}/{name}{i}.jpg", frame) # path created in add person function
                    print(f"Image {i} captured successfully")
                    play_machine_voice.play_machine_audio("Image_captured.mp3")
                    i += 1
                else:
                    print("Unclear image, please retake with proper lighting and with only 1 person")
                    play_machine_voice.play_machine_audio("Unclear Image.mp3")
                    
            if GPIO.input(448) == True:
                play_machine_voice.play_machine_audio("exit_button_pressed.mp3")
                print("Deleting taken images")
                delete_face(name)
                break
                

        cap.release()
        cv2.destroyAllWindows()
        
        for i in range(1, 4):
            image_path = os.path.join(path, f"{name}{i}.jpg")
            if not os.path.exists(image_path):
                return False
        return True
        
    
    def check_face(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.detector(rgb, model='hog')
        if len(result) == 1:
            return True
        else:
            return False
        
    def process_raw_imgs(self, path, name):
        encodings = []
        for img in os.listdir(path):
            image = Image.open(os.path.join(path, img))
            image = image.convert('RGB')
            pixels = np.asarray(image)
            result = self.detector(pixels, model='hog')
            
            if result and len(result) > 0:  # Check if result is not empty and has at least one face detected
                encoding = self.feature_extractor(pixels, result)
                if encoding and len(encoding) > 0:  # Check if encoding is not empty and has at least one encoding
                    encodings.append(encoding[0])
                    aug_imgs = self.augument(pixels)
                    for aug_img in aug_imgs:
                        encoding = self.feature_extractor(aug_img, result)
                        if encoding and len(encoding) > 0:  # Check if encoding is not empty and has at least one encoding
                            encodings.append(encoding[0])

                        
        encodings = np.asarray(encodings)
        print(encodings)
        encodings = self.normalizer.transform(encodings)
        print(encodings.shape)
        np.save(f"{EMBEDDINGS_DIR}/{name}.npy", encodings) # encoded file creation
        
    def build_model(self):
        encodings = []
        y = []
        i = 0
        label_dict = {}
        for emb in os.listdir(EMBEDDINGS_DIR):
            encodings += list(np.load(os.path.join(EMBEDDINGS_DIR, emb)))
            cls = emb.split('.')[0]
            y += [i] * NO_IMG_PER_PERSON * 6
            label_dict[i] = cls
            i += 1
        encodings = np.asarray(encodings)
        self.label_enc.fit(y)
        y = self.label_enc.transform(y)
        self.model.fit(encodings, y)
        
        with open(MAPPING_PATH, 'wb') as fp:
            pickle5.dump(label_dict, fp, protocol=pickle5.HIGHEST_PROTOCOL)
        pickle5.dump(self.model, open(RECOGNIZER_PATH, 'wb'))
        self.recognizer = self.model
        self.label_dict = label_dict
        self.persons = os.listdir(RAW_IMG_DIR)
        
        
    def add_person(self):
        if len(os.listdir(AUDIODIR)) >= 100:
            play_machine_voice.play_machine_audio("adding_face_limit_exceeded_please_delete_some_faces.mp3")
            play_machine_voice.play_machine_audio("total_files.mp3")
            length = len(os.listdir(AUDIODIR))
            play_machine_voice.play_machine_audio(f"number_{length}.mp3")
            return
        try:
            play_machine_voice.play_machine_audio("now_tell_your_name.mp3")
            name = get_name()
            audio_path = voice_rec(name)
            
            
            
            if os.path.exists(audio_path):
                
                play_machine_voice.play_machine_audio("person's_name_recorded_as.mp3")
                person_id = os.path.join(AUDIODIR, name+ '.wav')
                play_rec(person_id)
                play_machine_voice.play_machine_audio("successfully_recorded_person's_name.mp3")
                play_machine_voice.play_machine_audio("Thank You.mp3")
                
                img_dir = os.path.join(RAW_IMG_DIR, name)
                os.makedirs(img_dir)
                
                status = self.collect_images(img_dir, name)
                
                start_time = time.time()
                
                if status:
                    self.process_raw_imgs(img_dir, name)
                    self.persons = os.listdir(RAW_IMG_DIR)
                else:
                    print("images not stored properly")
                    play_machine_voice.play_machine_audio("face_not_registered_change_your_location_and_try_again_with_proper_lighting.mp3")
                    delete_face(name)
                    return

                    
            else:
                print("Audio file not exist, try again,... deleting created folders and files")
                play_machine_voice.play_machine_audio("face_not_registered_change_your_location_and_try_again_with_proper_lighting.mp3")
                delete_face(name)
                return
            

            if len(self.persons) > 1:
                self.build_model()
            
            path = os.path.join(EMBEDDINGS_DIR,(name + ".npy"))
            
            if os.path.exists(path):
                print("Encoding done")
                print(f"Successfully added person {name}")
            
            play_machine_voice.play_machine_audio("Successfully added face.mp3")
        
            play_machine_voice.play_machine_audio("total face.mp3")
        
            length = len(os.listdir(AUDIODIR))
            play_machine_voice.play_machine_audio(f"number_{length}.mp3")
            return
            
        except Exception as e:
            print(e)
            delete_face(name)
            play_machine_voice.play_machine_audio("face_not_registered_change_your_location_and_try_again_with_proper_lighting.mp3")
            print("Person Files deleted")
            return
        
        if self.VERBOSE:
            print(f"Time taken for adding a person took {time.time() - start_time} seconds")

        else:
            print("Encoding failed")
            delete_face(name)
    
    def get_threshold(self):
        thres = THRESHOLD_BASE
        for i in range(5, len(self.persons), 5):
            thres -= 1
        return thres/100
    

    def recognize(self):
        if len(os.listdir(AUDIODIR)) == 0:
            play_machine_voice.play_machine_audio("there_is_no_one_persons_face_to_recognize.mp3")
            return  # Exit the function if AUDIODIR is empty
        
        else:
            if os.path.exists("/home/rock/Desktop/Hearsight/English/set_face_who/2.jpg"):
                os.remove("/home/rock/Desktop/Hearsight/English/set_face_who/2.jpg")
            a="ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video1 -frames 1 /home/rock/Desktop/Hearsight/English/set_face_who/2.jpg"
            THRESHOLD = self.get_threshold()
            
            while True:

                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                else:
                    b=os.system(a)
                    frame = cv2.imread("/home/rock/Desktop/Hearsight/English/set_face_who/2.jpg")
                    start_time = time.time()
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.detector(rgb, model='hog')
                    print("No of faces detected:", len(results))
                    if len(results) == 0:
                        play_machine_voice.play_machine_audio("Face not recognized.mp3")
                    print(f"Time taken to recognize: {time.time()-start_time} seconds")
                    rec = 0
                    
                    for result in results:
                        new_embed = self.feature_extractor(rgb, [result])
                        x1, x2, y1, y2 = result
                        print(self.recognizer)
                        
                        new_embed = self.normalizer.transform(new_embed)
                        y_pred = self.recognizer.predict_proba(new_embed)[0]
                        confidence = np.max(y_pred)
                        if confidence >= THRESHOLD:
                            person = self.label_dict[np.argmax(y_pred)]
                            print(person, int(confidence * 100))
                            rec += 1
                            cv2.putText(frame, f"{person}-{int(confidence * 100)}%", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=3)

                            file_search.find((person+'.wav'),AUDIODIR)
                            time.sleep(2)

                        print("Recognized person:", rec)
                        
                    if len(results) > rec:
                        for i in range(len(results) - rec):
                            play_machine_voice.play_machine_audio("unknown face.mp3")
                    cv2.destroyAllWindows()
                    
                    print(f"Recognition + detection time of all faces {time.time() - start_time}")
                    time.sleep(0.1)
                    cv2.destroyAllWindows()
                    os.remove("/home/rock/Desktop/Hearsight/English/set_face_who/2.jpg")
                    break
                
    def remove_person(self):
        img_folders = os.listdir(RAW_IMG_DIR)
        img_folders = sorted(img_folders, reverse=True)
        print(img_folders)
        he = len(img_folders)
        print(he)
        she = str(img_folders)[1:-1]
        print(she)

        if len(os.listdir(AUDIODIR)) == 0:
            play_machine_voice.play_machine_audio("no face to remove.mp3")
            return  # Exit the function if AUDIODIR is empty

        play_machine_voice.play_machine_audio("press_feature_button.mp3")

        count = -1
        while True:
            time.sleep(0.7)
            if GPIO.input(450) == True:
                count += 1
                if count >= he:
                    count = 0
                print(count)
                print(img_folders[count])
                person_id = os.path.join(AUDIODIR, img_folders[count] + '.wav')
                play_rec(person_id)
                play_machine_voice.play_machine_audio("now_press_confirm_button_to_delete_this_persons_face.mp3")
                play_machine_voice.play_machine_audio("otherwise.mp3")
                play_machine_voice.play_machine_audio("press your feature button now.mp3")

            if GPIO.input(421) == True:
                count -= 1
                if count < -he:
                    count = -1
                print(count)
                print(img_folders[count])
                person_id = os.path.join(AUDIODIR, img_folders[count] + '.wav')
                play_rec(person_id)
                play_machine_voice.play_machine_audio("now_press_confirm_button_to_delete_this_persons_face.mp3")
                play_machine_voice.play_machine_audio("otherwise.mp3")
                play_machine_voice.play_machine_audio("press your feature button now.mp3")
                
            if GPIO.input(447) == True:
                play_machine_voice.play_machine_audio("Confirm Button Pressed.mp3")
                delete_face(img_folders[count])
                play_machine_voice.play_machine_audio("Successfully removed face.mp3")
                break

            if GPIO.input(448) == True:
                play_machine_voice.play_machine_audio("exit_button_pressed.mp3")
                break
        
recognizer = Recognizer()