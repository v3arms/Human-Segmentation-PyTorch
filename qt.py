import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QDialogButtonBox
from PyQt5.QtWidgets import QProgressBar, QFileDialog, QWidget, QTextEdit, QErrorMessage, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QThread, QEventLoop, pyqtSignal
from base.base_inference import VideoInference
from models import DeepLabV3Plus
from os import path 
import torch
import ffmpeg


CHECKPOINT = "./pretr/model.pth"
BACKBONE   = "resnet18"

'''
class timerBarThread(QThread) :
    def __init__(self, bar):
        QThread.__init__(self)
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.setInterval(250)
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.updatePB)
        self.bar = bar
    
    def start(self) :
        self.timer.start()
        # loop = QEventLoop()
        # loop.exec_()
    
    def updatePB(self) :
        print("here!")
        if self.inference is not None :
            self.bar.setValue(self.inference.cur_frame * 100 // self.inference.fr_count)    
'''


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('ui.ui', self)
        
        self.openBt      = self.findChild(QPushButton, 'openBt')
        self.runBt       = self.findChild(QPushButton, 'runBt')

        self.openBt.clicked.connect(self.openButtonClicked)
        self.runBt.clicked.connect(self.runButtonClicked)

        self.v_in      = None
        self.v_out     = None
        self.inference = None

        print("Loading model")
        self.model = DeepLabV3Plus(backbone=BACKBONE, num_classes=2)
        trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
        self.model.load_state_dict(trained_dict, strict=False)
        if torch.cuda.is_available() :
            self.model.cuda()
        self.model.eval()
        print('Complete.')

        self.show()
        
        
    def openButtonClicked(self) :
        self.v_in, _ = QFileDialog.getOpenFileName(self, "Select video", options=QFileDialog.DontUseNativeDialog)

        if self.v_in:
            self.v_out = path.splitext(self.v_in)[0] + '_out' + path.splitext(self.v_in)[1]
            print(self.v_out)
        
    
    def runButtonClicked(self) :
        if self.v_out is None or self.v_in is None :
            msg = QMessageBox(self)
            msg.setText("Error : no video specified.")
            msg.setWindowTitle("oops")
            msg.showNormal()
            return
        cuda = False
        if torch.cuda.is_available() :
            cuda = True

        print("CUDA : ", cuda)
        self.inference = VideoInference(
            model=self.model,
            video_path=self.v_in,
            video_out_path='./.tmp.mp4',
            input_size=320,
            background_path = "./backgrounds/white.jpg",
            use_cuda=cuda,
            draw_mode='matting'
        )
        print('Start processing frames...')
        self.inference.run()
        print('Done.')

        print('Running ffmpeg to merge video channels...')
        in1 = ffmpeg.input(self.v_in)
        in2 = ffmpeg.input('./.tmp.mp4')
        out = ffmpeg.output(in1.audio, in2.video, self.v_out)
        out.run(overwrite_output=True)
        print('All done.')

        msg = QMessageBox(self)
        msg.setText("Complete!")
        msg.setWindowTitle("status")
        msg.showNormal()



def main() :
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
