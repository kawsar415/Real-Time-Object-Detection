import cv2
#img = cv2.imread('man.PNG')

thres = 0.45
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weigthpath, configPath)
net.setInputSize(320, 320)
net.setInputScale((1.0/127.5))
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    suceess, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold = thres)
    print(classIds, bbox)

    # for classID in classIDs
    if len(classIds) != 0:
        for classId, confidence, box in zip (classIds.flatten(), confs.flatten(),bbox):
            cv2.rectangle(img, box, color=(0, 255, 0),thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255,0),2)
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255,0),2)


    cv2.imshow("Output",img)
    cv2.waitKey(1)




# import cv2
# img = cv2.imread('man.PNG')

# thres = 0.45
# # cap = cv2.VideoCapture(1)
# # cap.set(3, 640)
# # cap.set(4, 480)

# classNames = []
# classFile = 'coco.names'
# with open(classFile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')
# # print(classNames)
# configPath = 'RealTimeobjectdetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weigthpath = 'RealTimeobjectdetection/frozen_inference_graph.pb'

# net = cv2.dnn_DetectionModel(weigthpath, configPath)
# # net = cv2.dnn.readNetFromTensorflow(weightPath,configPath)
# net.setInputSize(320, 320)
# net.setInputScale((1.0/127.5))
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)
# classIds, confs, bbox = net.detect(img, confThreshold = thres)
# print(classIds, bbox)

# # while True:
# #     suceess, img = cap.read()
#     # classIds, confs, bbox = net.detect(img, confThreshold = thres)
#     # print(classIds, bbox)

# #     # for classID in classIDs
# #     if len(classIds) != 0:
# #         for classId, confidence, box in zip (classIds.flatten(), confs.flatten(),bbox):
# #             cv2.rectangle(img, box, color=(0, 255, 0),thickness=2)
# #             cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255,0),2)
# #             cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255,0),2)


# cv2.imshow("Output",img)
# cv2.waitKey(0)












