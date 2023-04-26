# encoding=utf-8
import pickle
from ui.new_main import *
from PyQt5.QtWidgets import QMainWindow
from drawroi import MyLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.Qt import QButtonGroup
class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setFixedSize(1004, 744)
        self.setupUi(self)
        self.retranslateUi(self)
        # 加载图片
        self.label=MyLabel(self.label)
        # print(self.label.width())
        # print(self.label.height())
        self.image = QPixmap("1.jpg")
        self.label.setScaledContents(True)
        self.label.setPixmap(self.image)
        # print(self.label.width())
        # print(self.label.height())
        self.label.setCursor(Qt.CrossCursor)
        self.label.dialogSignel.connect(self.showroi)

        #单选框
        self.trige_select=QButtonGroup(self)
        self.trige_select.addButton(self.radioButton_5)
        self.trige_select.addButton(self.radioButton_6)

        self.hardware_select=QButtonGroup(self)
        self.hardware_select.addButton(self.radioButton)
        self.hardware_select.addButton(self.radioButton_2)

        self.output_select=QButtonGroup(self)
        self.output_select.addButton(self.radioButton_3)
        self.output_select.addButton(self.radioButton_4)

    def showroi(self,ret):
        res=ret.getRect()
        self.lineEdit_12.setText(str(res[0]))
        self.lineEdit_13.setText(str(res[1]))
        self.lineEdit_14.setText(str(res[2]))
        self.lineEdit_15.setText(str(res[3]))







