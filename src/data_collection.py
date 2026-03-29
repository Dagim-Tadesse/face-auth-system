import cv2
import os
user_Name=input("Please Enter Eour name ? ").strip()
cap=cv2.VideoCapture(0)
save_directory=os.path.join("face-auth-system","src","data","row",user_Name)
os.makedirs(save_directory,exist_ok=True)
count =0
max_no_image=15
while True:
    ref,imageframe=cap.read()
    
    if not ref: 
        print ( 'the image is not working ')
    cv2.imshow("Webcame",imageframe)
    keytype=cv2.waitKey(1)
    if keytype==27:
        break
    if keytype==32:
        if count <max_no_image:   
            filename=os.path.join(save_directory,f"image_of_{user_Name}_{count+1}.jpg" )         
            cv2.imwrite(filename,imageframe)
            count=count+1
        else :
             break
cap.release()
