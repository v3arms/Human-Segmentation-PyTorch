import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QProgressBar, QFileDialog, QWidget, QTextEdit
from PyQt5 import uic
from base.base_inference import VideoInference
from models import UNet
from models import DeepLabV3Plus
from subprocess import call
from os import path
from 
import torch
import ffmpeg


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('ui.ui', self)
        self.show()
        
        self.openBt   = self.findChild(QPushButton, 'openBt')
        self.runBt    = self.findChild(QPushButton, 'runBt')
        self.textEdit = self.findChild(QTextEdit, 'textEdit')
        
        self.openBt.clicked.connect(self.openButtonClicked)
        self.runBt.clicked.connect(self.runButtonClicked)
        
        
    def openButtonClicked(self) :
        self.v_in, _ = QFileDialog.getOpenFileName(self, "Select video")
        if self.v_in:
            self.v_out = 
        
    
    def runButtonClicked(self) :
        pass
 
 
    def updatePB(self) :
        pass
    
        
    def updateTE(self) :
        pass




app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()