import cv2
import darknet.darknet as darknet
import time


win_title = 'YOLOv4 CUSTOM DETECTOR'
cfg_file = 'config/yolov4-tiny-mask.cfg'
data_file = 'config/opencv.data'
weight_file = 'darknet/backup/yolov4-tiny-mask_last.weights'
thre = 0.25
show_coordinates = True

network, class_names, class_colors = darknet.load_network(
        cfg_file,
        data_file,
        weight_file,
        batch_size=1
    )

width = 720
height = 480

#width = darknet.network_width(network)
#height = darknet.network_height(network)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_prev = time.time()
    
    frame_rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize( frame_rgb, (width, height))
    
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) 
    
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)
    
    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fps = int(1/(time.time()-t_prev))
    cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
    cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(win_title, image)
    
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
