from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
# from PyQt5.QtCore import QThread, pyqtSignal

import time
import os


#from .UI import Ui_MainWindow
from .player_tracking import Ui_MainWindow
from .video_controller import video_controller
class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.comboPathList = []

    def setup_control(self):
        self.ui.button_openFile.clicked.connect(self.open_file)        

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path
        self.video_path = filename
        self.ui.comboBox.clear()
        self.video_controller = video_controller(video_path=self.video_path, ui=self.ui, comboPathList=self.comboPathList)

        i = self.video_path.rfind('/')
        originfilename = self.video_path[i+1:]
        self.ui.comboBox.addItem(f'{originfilename}')
        self.comboPathList.append(f'{filename}')

        self.ui.label_filePath.setText(self.video_path)
        self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
        self.ui.button_stop.clicked.connect(self.video_controller.stop)
        self.ui.button_pause.clicked.connect(self.video_controller.pause)
        self.ui.button_startTrack.clicked.connect(self.video_controller.tracking)
        self.ui.button_boundary.clicked.connect(self.video_controller.boundary)
        self.ui.button_clipFile.clicked.connect(self.video_controller.clip)
        self.ui.comboBox.currentIndexChanged.connect(self.video_controller.load_file)