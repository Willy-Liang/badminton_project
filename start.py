from PyQt5 import QtWidgets

from GUI.controller import MainWindow_controller

if __name__ == '__main__':
    '''GUI'''
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())