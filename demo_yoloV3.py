import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classFile = "coco.names"
classnames = []
with open(classFile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)

modelconfig = "yoloV3.cfg"
modelweights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(cfgFile=modelconfig,darknetModel=modelweights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img,threshold_conf):
    ht, wt, ch = img.shape
    bbox_list = []
    classIds_list = []
    confs_list = []

    for output in outputs:
        for det in output:
            probabilities = det[5:]
            class_id = np.argmax(probabilities)
            confidence = probabilities[class_id]
            if confidence > threshold_conf:
                w,h = int(det[2]*wt),int(det[3]*ht)
                x,y = int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
                bbox_list.append([x,y,w,h])
                classIds_list.append(class_id)
                confs_list.append(float(confidence))

    print(bbox_list)
    main_box_indices = cv2.dnn.NMSBoxes(bbox_list,confs_list,threshold_conf,0.3)
    for index in main_box_indices:
        x,y,w,h = bbox_list[index[0]][0],bbox_list[index[0]][1],bbox_list[index[0]][2],bbox_list[index[0]][3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),thickness=1)
        cv2.putText(img,f"{classnames[classIds_list[index[0]]].upper()} {int(confs_list[index[0]] * 100)}%",
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,scalefactor=1/255,size=(320,320),mean=[0,0,0],swapRB=1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputLayerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputLayerNames)

    outputs = net.forward(outputLayerNames)

    findObjects(outputs,img,0.5)


    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Detection",window_width, window_height,)
    cv2.imshow("Detection",img)
    cv2.waitKey(1)