from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from mainWindow_ui import Ui_MainWindow
from imageClassifier import ImageClassifier


class classifcationResults(object):
    def __init__(self,
                 image_indx=0,
                 labels_string=[],
                 labels_score=[]):
        self.image_indx = image_indx
        self.labels_string = labels_string
        self.labels_score = labels_score

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        # Parameters
        self.imagefiles = ''
        self.curImageIndx = 0
        self.totalImageNum = 0
        self.results = []
        self.imageclassifer = ImageClassifier()

        # UI stuff
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionOpen.triggered.connect(self.openFileNamesDialog)
        self.ui.pushButton_right.clicked.connect(self.nextImage)
        self.ui.pushButton_left.clicked.connect(self.preImage)

    def showImages(self, filename):
        pixmap = QPixmap(filename)
        #pixmap = pixmap.scaled(self.ui.label_image.width(), self.ui.label_image.height(), Qt.KeepAspectRatio)
        self.ui.label_image.setPixmap(pixmap)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Images (*.jpg *.jpeg *.png *.bmp)", options=options)
        if fileName:
            print(fileName)
            self.imagefiles = fileName
            self.showImages(fileName)

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "Images (*.jpg *.jpeg *.png *.bmp)", options=options)
        if files:
            print(files)
            self.imagefiles = files
            self.showImages(files[0])

            self.curImageIndx = 0
            self.totalImageNum = len(files)
            self.ui.label_imageIndx.setText('{0}/{1}'.format(self.curImageIndx+1, self.totalImageNum))

            self.imageClassification(files[0])
            self.show_results()

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)

    def nextImage(self):
        self.curImageIndx += 1
        if self.curImageIndx > self.totalImageNum-1:
            self.curImageIndx = 0

        imagefile = self.imagefiles[self.curImageIndx]
        self.ui.label_imageIndx.setText('{0}/{1}'.format(self.curImageIndx+1, self.totalImageNum))
        self.showImages(imagefile)

        self.imageClassification(imagefile)
        self.show_results()


    def preImage(self):
        self.curImageIndx -= 1
        if self.curImageIndx < 0:
            self.curImageIndx = self.totalImageNum-1

        imagefile = self.imagefiles[self.curImageIndx]
        self.ui.label_imageIndx.setText('{0}/{1}'.format(self.curImageIndx+1, self.totalImageNum))
        self.showImages(imagefile)

        self.imageClassification(imagefile)
        self.show_results()

    def imageClassification(self, imagefile):
        result = next((item for item in self.results if item['index'] == self.curImageIndx), False)
        if not result:
            result = {}
            result['index'] = self.curImageIndx
            result['filename'] = imagefile
            result['prediction'] = self.imageclassifer.run_inference_on_image(imagefile)
            self.results.append(result)

    def show_results(self):
        result = next((item for item in self.results if item['index'] == self.curImageIndx), False)
        print result
        if result:
            labels = result['prediction']['top5_labels']
            scores = result['prediction']['top5_scores']
            self.ui.label_result_1.setText(labels[0])
            self.ui.label_result_2.setText(labels[1])
            self.ui.label_result_3.setText(labels[2])
            self.ui.label_result_4.setText(labels[3])
            self.ui.label_result_5.setText(labels[4])

            self.ui.progressBar_1.setValue(int(scores[0] * 100))
            self.ui.progressBar_2.setValue(int(scores[1] * 100))
            self.ui.progressBar_3.setValue(int(scores[2] * 100))
            self.ui.progressBar_4.setValue(int(scores[3] * 100))
            self.ui.progressBar_5.setValue(int(scores[4] * 100))

