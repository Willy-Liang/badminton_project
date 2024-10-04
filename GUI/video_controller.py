from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import QTimer, Qt, QRect, QThread, pyqtSignal

from .opencv_engine import opencv_engine

import os
import sys
import argparse
import ast
import cv2
import time
import torch
from torch.utils.data import DataLoader
from vidgear.gears import CamGear
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

import csv

from TrackNetV3.test import predict_location, get_ensemble_weight, generate_inpaint_mask
from TrackNetV3.dataset import Shuttlecock_Trajectory_Dataset
from TrackNetV3.utils.general import *

Threshold = 0.8

#
import pandas as pd
#

# videoplayer_state_dict = {
#  "stop":0,   
#  "play":1,
#  "pause":2     
# }

class video_controller(object):
    def __init__(self, video_path, ui, comboPathList):
        self.video_path = video_path
        self.comboPathList = comboPathList
        # modify
        #self.v3_data = None
        #self.v3_data_path = v2_data_path
        #self.v3_data = pd.read_csv(v3_data_path)
        self.listBallCounter = None
        self.ui = ui
        self.qpixmap_fix_width = 960 # 16x9 = 1920x1080 = 1280x720 = 800x450 = 960X540
        self.qpixmap_fix_height = 540
        self.current_frame_no = 0
        self.videoplayer_state = "pause"
        self.init_video_info()
        self.set_video_player()
    
    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"]
        self.video_fps = videoinfo["fps"]
        self.video_total_frame_count = videoinfo["frame_count"]
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"]

        self.x_rate = self.video_width/960
        self.y_rate = self.video_height/540
        self.boundary_width = self.video_width
        self.boundary_height = self.video_height
        self.boundary_corner = [0, 0]
        self.set_up_boundary = False
        self.boundary_cnt = 0
        self.boundary_1 = [0, 0]
        self.boundary_2 = [960, 540]

        self.ui.slider_videoframe.setRange(0, self.video_total_frame_count-1)
        self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)
        self.ui.label_videoPlayer.mousePressEvent = self.show_mouse_press

    def show_mouse_press(self, event):
        if self.set_up_boundary:
            self.boundary_cnt += 1
            self.ui.label_state.setText("請設定邊界右下角")
            x = event.pos().x()
            y = event.pos().y()
            self.__update_text_clicked_position(x, y)
            if self.boundary_cnt == 2:
                self.boundary_cnt = 0                
                self.ui.label_state.setText("")
                self.set_up_boundary = False                
                self.boundary_width = int((self.boundary_2[0] - self.boundary_1[0])*self.x_rate)
                self.boundary_height = int((self.boundary_2[1] - self.boundary_1[1])*self.y_rate)
                
    def boundary(self):
        self.set_up_boundary = not self.set_up_boundary
        self.boundary_cnt = 0
        self.ui.label_state.setText("請設定邊界左上角")

    def __update_text_clicked_position(self, x, y):
        if self.boundary_cnt == 1:
            self.boundary_1 = [x, y]
            self.boundary_corner[0] = int(self.boundary_1[0]*self.x_rate)
            self.boundary_corner[1] = int(self.boundary_1[1]*self.y_rate)
        elif self.boundary_cnt == 2:
            self.boundary_2 = [x, y]
    
    # def clip(self):
    #     self.timer.stop()
    #     i = self.video_path.rfind('/')
    #     filename = self.video_path[i+1:]
    #     if filename is not None:
    #         video = cv2.VideoCapture(filename)
    #         assert video.isOpened()
    #     t_start = time.time()
    #     video_writer = None
    #     videoCounter = 0
    #     while True:
    #         if filename is not None:
    #             ret, frame = video.read()
    #             # framecounter = framecounter + 1
    #             # self.ui.progressBar.setValue((int)(framecounter+1)*100/self.video_total_frame_count)
    #             if not ret:
    #                 t_end = time.time()
    #                 print("\n Total Time: ", t_end - t_start)
    #                 break
    #             hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #             lower_color = np.array([35, 43, 46])
    #             upper_color = np.array([77, 255, 255])
    #             mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    #             pixel_count = cv2.countNonZero(mask)
    #             if (pixel_count/(frame.shape[1]*frame.shape[0])) > Threshold:
    #                 if video_writer is None:
    #                     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video format
    #                     video_writer = cv2.VideoWriter(f'clipOutPut/clip{videoCounter}.mp4', fourcc, self.video_fps, (frame.shape[1], frame.shape[0]))
    #                 video_writer.write(frame)
    #             else:
    #                 if video_writer is not None:
    #                     video_writer.release()
    #                     video_writer = None
    #                     videoCounter += 1
    #         else:
    #             frame = video.read()
    #             if frame is None:
    #                 video.release()
    #                 break
    #     video_writer.release()

    def clip(self):
        self.timer.stop()
        self.ui.label_state.setText("影片剪輯中...")
        i = self.video_path.rfind('/')
        filename = self.video_path[i+1:]
        if filename is not None:
            video = cv2.VideoCapture(filename)
            assert video.isOpened()
        # t_start = time.time()
        framecounter = 0
        isInClip = False
        clipStart = 0
        clipEnd = 0
        clips = []
        while True:
            if filename is not None:
                ret, frame = video.read()
                framecounter = framecounter + 1
                self.ui.progressBar.setValue((int)(framecounter+1)*100/self.video_total_frame_count)
                if not ret:
                    # t_end = time.time()
                    # print("\n Total Time: ", t_end - t_start)
                    break
                crop_img = frame[self.boundary_corner[1]:self.boundary_corner[1]+self.boundary_height, self.boundary_corner[0]:self.boundary_corner[0]+self.boundary_width]
                hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                lower_color = np.array([35, 43, 46])
                upper_color = np.array([77, 255, 255])
                mask = cv2.inRange(hsv_frame, lower_color, upper_color)
                pixel_count = cv2.countNonZero(mask)
                if (pixel_count/(crop_img.shape[1]*crop_img.shape[0])) > Threshold:
                    if isInClip is False:
                        isInClip = True
                        clipStart = framecounter
                else:
                    if isInClip is True:
                        isInClip = False
                        clipEnd = framecounter
                        if clipEnd - clipStart > 3*self.video_fps: #大於3秒
                            print(f"{clipStart},{clipEnd}")
                            clipsH = clipStart-self.video_fps
                            if clipsH < 0:
                                clipsH = 0
                            clipsT = clipEnd+self.video_fps
                            if clipsT > self.video_total_frame_count:
                                clipsT = self.video_total_frame_count
                            if len(clips) == 0:                                
                                clips.append([clipsH ,clipsT])
                            else:
                                if clips[-1][1] > clipsH:
                                    clipsH = clips[-1][1] + 1
                                clips.append([clipsH ,clipsT])
            else:
                frame = video.read()
                if frame is None:
                    video.release()
                    break
        self.writeClips(clips=clips)
        print('done')
        self.ui.label_state.setText("影片剪輯完成!")
        self.set_video_player()
    
    def writeClips(self, clips):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        i = self.video_path.rfind('/')
        filename = self.video_path[i+1:]
        if filename is not None:
            video = cv2.VideoCapture(filename)
            assert video.isOpened()
        framecounter = 0
        videoNumber = 0
        while len(clips) > 0:
            clipsH = clips[0][0]
            clipsT = clips[0][1]
            del clips[0]
            videoWriter = cv2.VideoWriter(f'clipOutPut/clip{videoNumber}.mp4', fourcc, self.video_fps, (self.video_width,  self.video_height))
            self.ui.comboBox.addItem(f'clip{videoNumber}.mp4')
            trueFilePath = os.getcwd() + f'/clipOutPut/clip{videoNumber}.mp4'
            self.comboPathList.append(trueFilePath)
            videoNumber += 1
            while True:
                ret, frame = video.read()
                framecounter += 1
                if not ret:
                    break
                if framecounter < clipsH:
                    continue
                elif framecounter > clipsT:
                    break
                else:
                    videoWriter.write(frame)           
            print(f"{clipsH},{clipsT}")
        videoWriter.release()
        video.release()

    def load_file(self):
        num = self.ui.comboBox.currentIndex()
        self.video_path = self.comboPathList[num]
        self.ui.label_filePath.setText(self.video_path)
        self.init_video_info()

    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        # self.timer.start(1000//self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)

    def set_current_frame_no(self, frame_no):
        self.vc.set(1, frame_no) # bottleneck

    def __get_next_frame(self):
        ret, frame = self.vc.read()
        self.ui.label_frameInfo.setText(f"frame number: {self.current_frame_no}/{self.video_total_frame_count}")
        self.setslidervalue(self.current_frame_no)
        if self.listBallCounter is not None:
            self.ui.label_hits.setText(f"hits:{self.listBallCounter[self.current_frame_no+1][4]}") 
        return frame

    def __update_label_frame(self, frame):       
        bytesPerline = 3 * self.video_width
        frame = opencv_engine.draw_point(frame, point=(self.boundary_corner[0], self.boundary_corner[1]))
        if self.boundary_2 != [960, 540]:
            frame = opencv_engine.draw_rectangle_by_xywh(frame, xywh=(self.boundary_corner[0], self.boundary_corner[1], self.boundary_width, self.boundary_height))
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.label_videoPlayer.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.label_videoPlayer.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center

        # # modify
        # self.qpixmap2 = self.qpixmap.copy()
        
        # if (self.v3_data != None):
        #     if (self.v3_data.at[self.current_frame_no+1, 'Visibility']):
        #         painter = QPainter()
        #         painter.begin(self.qpixmap)
        #         painter.setPen(QColor(255, 0, 0))
        #         #x = round(self.v2_data.at[self.current_frame_no+1, 'X']/2)
        #         #y = round(self.v2_data.at[self.current_frame_no+1, 'Y']/2)
        #         x = round(self.v3_data.at[self.current_frame_no+1, 'X']/1.5)
        #         y = round(self.v3_data.at[self.current_frame_no+1, 'Y']/1.5)
        #         painter.drawRect(x-5, y-5, 10, 10)
        #         painter.end()
        # #

    def play(self):
        self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"

    def timer_timeout_job(self):
        if (self.videoplayer_state == "play"):
            if self.current_frame_no >= self.video_total_frame_count-1:
                #self.videoplayer_state = "pause"
                self.current_frame_no = 0 # auto replay
                self.set_current_frame_no(self.current_frame_no)
            else:
                self.current_frame_no += 1

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0
            self.set_current_frame_no(self.current_frame_no)

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
            self.set_current_frame_no(self.current_frame_no)

        frame = self.__get_next_frame()
        self.__update_label_frame(frame)

    def getslidervalue(self):
        self.current_frame_no = self.ui.slider_videoframe.value()
        self.set_current_frame_no(self.current_frame_no)

    def setslidervalue(self, value):
        self.ui.slider_videoframe.setValue(self.current_frame_no)

    def tracking(self):
        self.timer.stop()
        self.ui.label_state.setText("影片分析中...")
        time.sleep(1)
        predict_ball(video_file=self.video_path, tracknet_file='TrackNetV3/ckpts/TrackNet_best.pt', inpaintnet_file='TrackNetV3/ckpts/InpaintNet_best.pt', save_dir='.', output_video=True)
        # i = self.video_path.rfind('/')
        # file_name = self.video_path[i+1:]
        file_name = 'ball_result.mp4'
        Track(filename=file_name, save_video=True, video_framerate=self.video_fps, device='cuda', video_controller=self)
        self.video_path = os.getcwd() + "/output.avi"
        self.ui.label_filePath.setText(self.video_path)
        ballCounter = 'clip1_result.csv'
        with open(ballCounter) as csvFile :
            csvReader = csv.reader(csvFile)
            self.listBallCounter = list(csvReader)
        self.ui.label_state.setText("影片分析完成!")
        self.init_video_info()
        self.timer.start(1)
 
def Track(camera_id=0, filename=None, hrnet_m='HRNet', hrnet_c=48, hrnet_j=17, hrnet_weights="./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set="coco", image_resolution='(384, 288)',
         single_person=False, yolo_version="v3", use_tiny_yolo=False, disable_tracking=False, max_batch_size=16, disable_vidgear=False, save_video=False,
         video_format='MJPG', video_framerate=None, device=None, enable_tensorrt=False, video_controller=None):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    '''Camera'''
    # else:
    #     rotation_code = None
    #     if disable_vidgear:
    #         video = cv2.VideoCapture(camera_id)
    #         assert video.isOpened()
    #     else:
    #         video = CamGear(camera_id).start()

    if yolo_version == 'v3':
        if use_tiny_yolo:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3-tiny.weights"
        else:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3.weights"
        yolo_class_path = "./models_/detectors/yolo/data/coco.names"
    elif yolo_version == 'v5':
        # YOLOv5 comes in different sizes: n(ano), s(mall), m(edium), l(arge), x(large)
        if use_tiny_yolo:
            yolo_model_def = "yolov5n"  # this  is the nano version
        else:
            yolo_model_def = "yolov5m"  # this  is the medium version
        if enable_tensorrt:
            yolo_trt_filename = yolo_model_def + ".engine"
            if os.path.exists(yolo_trt_filename):
                yolo_model_def = yolo_trt_filename
        yolo_class_path = ""
        yolo_weights_path = ""
    else:
        raise ValueError('Unsopported YOLO version.')

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_version=yolo_version,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
        enable_tensorrt=enable_tensorrt
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0
    t_start = time.time()
    framecounter = 0
    file = open('points.csv', 'w',  newline = '')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Frame Number', 'Person ID', 'x', 'y', 'Confidence'])
    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                t_end = time.time()
                print("\n Total Time: ", t_end - t_start)
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break
        
        crop_img = frame[video_controller.boundary_corner[1]:video_controller.boundary_corner[1]+video_controller.boundary_height, video_controller.boundary_corner[0]:video_controller.boundary_corner[0]+video_controller.boundary_width]
        pts = model.predict(crop_img)
        
        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            for i in range(len(pt)):
                pt[i][0] += video_controller.boundary_corner[1]
                pt[i][1] += video_controller.boundary_corner[0]
                csv_writer.writerow([framecounter, pid, pt[i][1], pt[i][0], pt[i][2]])
            
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10)

        # for box in boxes:
        #     cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,255,255),2)

        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps, for %d person(s) ' % (fps,len(pts)), end='')
        framecounter = framecounter + 1
        video_controller.ui.progressBar.setValue((int)(framecounter+1)*100/video_controller.video_total_frame_count)

        if has_display:
            #cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    file.close()
    if save_video:
        video_writer.release()
        Temp_path = video_controller.video_path
        i = Temp_path.rfind('/')
        Temp_path = Temp_path[:i+1] + "output.avi"
        return Temp_path
    else:
        return "False"

def ball_prediction(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict

def predict_ball(video_file, tracknet_file, inpaintnet_file='', batch_size=1, eval_mode='weight', save_dir='pred_result', output_video=False, traj_len=8):
    num_workers = batch_size if batch_size <= 16 else 16
    # video_name = video_file.split('/')[-1][:-4]
    video_name = 'ball_result'
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if inpaintnet_file:
        inpaintnet_ckpt = torch.load(inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    # Sample all frames from video
    frame_list, fps, (w, h) = generate_frames(video_file)
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    print(f'Number of sampled frames: {len(frame_list)}')

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    if eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # Predict
            tmp_pred = ball_prediction(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        weight = get_ensemble_weight(seq_len, eval_mode)

        # Init prediction buffer params
        num_sample, sample_count = len(dataset), 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                    y_pred /= (sample_count+1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0)
                        y_pred /= (seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = ball_prediction(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = ball_prediction(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = ball_prediction(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]
        

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if output_video:
        write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, save_file=out_video_file, traj_len=traj_len)

    print('Done.')