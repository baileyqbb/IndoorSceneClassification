import sys
import imageViewer
from PyQt5.QtWidgets import *

def main():
    app = QApplication(sys.argv)
    application = imageViewer.ImageViewer()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()