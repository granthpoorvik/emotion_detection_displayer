import cv2
import threading
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from deepface import DeepFace
import emoji

# Define the emotion labels
emotion_labels = {
    'angry': 'angry',
    'disgust': 'disgusted',
    'fear': 'fearful',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

# Load the emoji icons
emoji_icons = {
    'angry': emoji.emojize('😡'),
    'disgusted': emoji.emojize('🥸'),
    'fearful': emoji.emojize('😨'),
    'happy': emoji.emojize('🤠'),
    'sad': emoji.emojize('😥'),
    'surprised': emoji.emojize('😲'),
    'neutral': emoji.emojize('😀')
}

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""random space"""
import random

def get_random_item(dictionary):
    key = random.choice(list(dictionary.keys()))
    value = dictionary[key]
    return key, value


#random_key, random_value = get_random_item(emoji_icons)
#print("Random Item:", random_key, "->", random_value)
random_emoji=get_random_item(emoji_icons)[1]


# Function to capture and display real-time video
def capture_and_display(random_emoji):
    count=0
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras
    width, height = 750, 400  # Adjust these values to suit your needs
    """the below comented lines are paced inside the loop and runed """
    #image = np.zeros((height, width, 3), np.uint8)     
    #pil_image = Image.fromarray(image)
    #draw = ImageDraw.Draw(pil_image)
    font_path = "C://Windows//Fonts//seguiemj.ttf"  # Replace with the actual font path
    font_size = 60
    x, y = 50, 70  # Adjust these coordinates as per your requirements
    
    while True:
        
        # Read frame from the video capture
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        emoji_icon=emoji.emojize('😀')# a default emoji
        # Check if any face is detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:

                # Extract the face region
                face = frame[y:y + h, x:x + w]

                # Perform emotion detection
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

                # Get the dominant emotion from the result list
                emotion = result[0]['dominant_emotion']

                # Print the emotion class with emoji
                emoji_icon = emoji_icons.get(emotion, '')
                print(f"Emotion: {emotion_labels[emotion]} {emoji_icon}")

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the emotion text near the face
                cv2.putText(frame, f" {emotion_labels[emotion]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                if random_emoji==emoji_icon:
                    count+=1
                    random_emoji=get_random_item(emoji_icons)[1]
                cv2.putText(frame, " emotion mimicked:-"+f" {count}", (30,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                
            
        
        # Display the frame in the first window
        cv2.imshow('Webcam for emotion detection', frame)
        
        """the below 3 lines are responsibe
        for the drawing of the image"""
        image = np.zeros((height, width, 3), np.uint8)   #change in code wher i have replaced the position of the line
        pil_image = Image.fromarray(image)               #the comented 3 lines are commented from
        draw = ImageDraw.Draw(pil_image)                 #
        
        """this is for random emotion generated window"""
       
        emoji_text = emoji.emojize(random_emoji*3)
        emoji_font = ImageFont.truetype(font_path, font_size)
        draw.text((x, y), emoji_text, font=emoji_font, fill=(255, 255, 255))
        image_rr = np.array(pil_image)
        cv2.imshow("RandomEmoji ", image_rr)
        
        """to capture the emoji in to the screen"""
        emoji_text = emoji.emojize(emoji_icon*3)  
        emoji_font = ImageFont.truetype(font_path, font_size)
        draw.text((x, y), emoji_text, font=emoji_font, fill=(255, 255, 255))
        image = np.array(pil_image)
        cv2.imshow("Emoji captured from user", image)
        image=None
        
        
        
        # Check for key press
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Create and start the thread
thread = threading.Thread(target=capture_and_display(random_emoji))
thread.start()

# Wait for the thread to finish
thread.join()
