import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QProgressBar, QFileDialog, QWidget, QTextEdit, QErrorMessage
from PyQt5 import uic
from base.base_inference import VideoInference
from models import DeepLabV3Plus
from os import path 
import torch
import ffmpeg


CHECKPOINT = "./pretr/model.pth"
BACKBONE   = "resnet18"


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('ui.ui', self)
        
        
        self.openBt   = self.findChild(QPushButton, 'openBt')
        self.runBt    = self.findChild(QPushButton, 'runBt')
        self.textEdit = self.findChild(QTextEdit, 'textEdit')

        print("Loading model")
        self.model = DeepLabV3Plus(backbone=BACKBONE, num_classes=2)
        trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
        self.model.load_state_dict(trained_dict, strict=False)
        self.model.cuda()
        self.model.eval()
        print('Complete.')

        
        self.openBt.clicked.connect(self.openButtonClicked)
        self.runBt.clicked.connect(self.runButtonClicked)

        self.show()
        
        
    def openButtonClicked(self) :
        self.v_in, _ = QFileDialog.getOpenFileName(self, "Select video", options=QFileDialog.DontUseNativeDialog)

        if self.v_in:
            self.v_out = path.splitext(self.v_in)[0] + '_out' + path.splitext(self.v_in)[1]
            print(self.v_out)
        
    
    def runButtonClicked(self) :
        if self.v_out is None or self.v_in is None :
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Error : no video specified.')
            return
        
        inference = VideoInference(
            model=self.model,
            video_path=self.v_in,
            video_out_path='./.tmp.mp4',
            input_size=320,
            background_path = "./backgrounds/white.jpg",
            use_cuda=True,
            draw_mode='matting'
        )
        print('Start processing frames...')
        inference.run()
        print('Done.')

        print('Running ffmpeg to merge video channels...')
        in1 = ffmpeg.input(self.v_in)
        in2 = ffmpeg.input('./.tmp.mp4')
        out = ffmpeg.output(in1.audio, in2.video, self.v_out)
        out.run(overwrite_output=True)
        print('All done.')



 
    def updatePB(self) :
        pass
    
        
    def updateTE(self) :
        pass




app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()