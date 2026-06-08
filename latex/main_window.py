import sys
import faulthandler
import signal
faulthandler.enable(True)

import numpy as np
import cv2
import os
import shutil
import time
from ultralytics import YOLO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QLabel, QFileDialog, 
    QPushButton, QSplitter, QStatusBar,
    QProgressBar, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QImageReader, QIcon
from PySide6.QtCore import Qt, QSize, QTimer, QSettings

from viewer import ImageViewer
from detector import Detector
from theme_manager import ThemeManager
from file_manager import FileManager
from gallery_manager import GalleryManager

THUMB_SIZE = 140


class MainWindow(QMainWindow):

    def init_window(self):
        self.setWindowIcon(QIcon("icon.png"))
        self.setWindowTitle("Potholer")
        self.resize(1300, 750)

    def init_settings(self):
        self.settings = QSettings("Potholer", "Detector")
        self.current_theme = self.settings.value("theme", "dark")

        self.results_dir = self.settings.value("results_dir")

        if not self.results_dir:
            self.results_dir = os.path.normpath(
                os.path.abspath("results")
            )

    def init_managers(self):
        self.theme_manager = ThemeManager(self)
        self.file_manager = FileManager(self)
        self.gallery = GalleryManager(self)

    def init_state(self):
        self.images = []

        self.result_pixmaps = []
        self.processed_paths = []
        self.per_image_detections = []
        self.total_detections = 0

        self.stop_requested = False
        self.loading_images = False
        self.processing = False
        self.show_input_images = True
        self.results_saved = False

        self.welcome_widget = None

    def init_detector(self):
        self.detector = Detector(
            YOLO("runs/detect/train5/weights/best.pt"),
            conf_threshold=0.25
        )

    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = file_menu.addAction("Открыть изображения")
        open_action.triggered.connect(self.file_manager.open_images)
        save_action = file_menu.addAction("Сохранить результаты")
        save_action.triggered.connect(self.save_results)
        open_processed_action = file_menu.addAction("Открыть обработанные результаты")
        open_processed_action.triggered.connect(self.open_processed_results)
        exit_action = file_menu.addAction("Выход")
        exit_action.triggered.connect(self.close)

        process_menu = menu_bar.addMenu("Обработка")
        self.detect_action = process_menu.addAction("Обнаружить объекты")
        self.detect_action.triggered.connect(self.detect_objects)
        conf_menu = process_menu.addMenu("Порог обнаружения")
        process_menu.addSeparator()
        results_menu = process_menu.addMenu("Временные результаты")
        self.current_dir_action = results_menu.addAction("Папка: не выбрана")
        self.current_dir_action.setEnabled(False)
        change_dir_action = results_menu.addAction("Изменить...")
        change_dir_action.triggered.connect(self.file_manager.select_results_folder)
        clear_dir_action = results_menu.addAction("Очистить")
        clear_dir_action.triggered.connect(self.file_manager.clear_results_folder)

        for conf in [0.25, 0.40, 0.50, 0.70]:
            action = conf_menu.addAction(str(conf))
            action.triggered.connect(lambda checked, c=conf: self.set_conf_threshold(c))

        view_menu = menu_bar.addMenu("Вид")
        self.toggle_input_action = view_menu.addAction("Показывать входные изображения")
        self.toggle_input_action.setCheckable(True)
        self.toggle_input_action.setChecked(True)
        self.toggle_input_action.triggered.connect(self.toggle_input_images)
        sort_menu = view_menu.addMenu("Сортировка результатов")
        sort_by_name = sort_menu.addAction("По имени")
        sort_by_count = sort_menu.addAction("По числу объектов")
        sort_by_name.triggered.connect(lambda: self.sort_results("name"))
        sort_by_count.triggered.connect(lambda: self.sort_results("count"))
        theme_menu = view_menu.addMenu("Тема")
        dark_theme_action = theme_menu.addAction("Тёмная")
        light_theme_action = theme_menu.addAction("Светлая")
        dark_theme_action.triggered.connect(lambda: self.theme_manager.set_theme("dark"))
        light_theme_action.triggered.connect(lambda: self.theme_manager.set_theme("light"))


    def init_wid(self):
        #центр
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        #лево
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_container = QWidget()
        self.left_scroll.setWidget(self.left_container)
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.addWidget(QLabel("Входные изображения"))
        self.left_scroll.setFixedWidth(320)


        #право
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_container = QWidget()
        self.right_scroll.setWidget(self.right_container)
        self.right_layout = QVBoxLayout(self.right_container)
        self.result_grid = QGridLayout()
        self.right_layout.addLayout(self.result_grid)
        #self.right_layout.addStretch()

        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.gallery.update_result_grid)

        #сплиттер
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_scroll)
        splitter.addWidget(self.right_scroll)
        splitter.setSizes([300, 1000])
        main_layout.addWidget(splitter)

    def init_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.input_counter = QLabel("Вход: 0 ")
        self.processed_counter = QLabel("Обработано: 0 ")
        self.detection_counter = QLabel("Объектов найдено: 0 ")
        self.avg_counter = QLabel("Среднее: 0.00 ")
        self.time_counter = QLabel("Время: 0.00 сек ")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setValue(0)

        self.status_text = QLabel("") #сообщ о состоянии

        self.status.addWidget(self.input_counter)
        self.status.addWidget(self.progress_bar)
        self.status.addWidget(self.status_text)

        self.status.addPermanentWidget(self.processed_counter)
        self.status.addPermanentWidget(self.detection_counter)
        self.status.addPermanentWidget(self.avg_counter)
        self.status.addPermanentWidget(self.time_counter)


    def __init__(self):
        super().__init__()


        self.init_window()
        self.init_settings()
        self.init_state()
        self.init_managers()
        self.init_detector()
        self.init_menu()
        self.init_wid()
        self.init_statusbar()

        self.show_welcome_screen()
        self.update_results_dir_display()
        self.theme_manager.set_theme(self.current_theme)


    def update_progress(self, current, total, text=None):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

        if text:
            self.status_text.setText(text)

        QApplication.processEvents()

#приветственное
    def show_welcome_screen(self):
        self.clear_welcome_screen()

        self.welcome_widget = QWidget()
        layout = QVBoxLayout(self.welcome_widget)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Potholer")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
        """)

        info = QLabel(
            "Программа обнаружения дефектов дорожного покрытия\n\n"
            "1. Выберите файлы с изображениями.\n"
            "2. Установите порог обнаружения.\n"
            "3. Запустите обработку.\n"
            "4. Сохраните результаты.\n"

            "Также вы можете самостоятельно выбрать папку\nдля хранения временных данных в меню \"Обработка\"."
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 14px;")

        open_btn = QPushButton("Открыть изображения")
        detect_btn = QPushButton("Обнаружить объекты")

        open_btn.setMinimumHeight(50)
        detect_btn.setMinimumHeight(50)

        open_btn.clicked.connect(self.file_manager.open_images)
        detect_btn.clicked.connect(self.detect_objects)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(info)
        layout.addSpacing(30)
        layout.addWidget(open_btn)
        layout.addWidget(detect_btn)

        self.right_layout.addStretch()
        self.right_layout.addWidget(self.welcome_widget)
        self.right_layout.addStretch()

    def clear_welcome_screen(self):
        if self.welcome_widget:
            self.welcome_widget.setParent(None)
            self.welcome_widget = None



    def update_results_dir_display(self):
        path = self.results_dir

        if not path:
            text = "Папка не выбрана"
        else:
            short = path
            if len(short) > 45:
                short = "..." + short[-42:]
            text = f"Папка: {short}"

        self.current_dir_action.setText(text)


    def set_images(self, files):
        self.images = files
        self.input_counter.setText(f"Вход: {len(files)} ")

        if self.show_input_images:
            self.gallery.show_thumbnails()
        else:
            self.status_text.setText("Загрузка изображений...")

            for idx, _ in enumerate(files):
                self.update_progress(idx + 1, len(files))

            self.status_text.setText("Загрузка завершена")

    def save_results(self):
        result = self.file_manager.save_results()

        if result is False:
            self.status_text.setText("Сохранение отменено")
            return

        if result is None:
            self.status_text.setText("Ошибка при сохранении отчёта")
            return

        self.results_saved = True
        self.status_text.setText(
            f"Сохранено: {result} изображений + отчёт"
        )
    


    def open_processed_results(self):

        data = self.file_manager.open_processed_results()

        if not data:
            self.status_text.setText("Ошибка загрузки результатов")
            return

        self.clear_welcome_screen()

        self.processed_paths = data["paths"]
        self.per_image_detections = data["detections"]

        self.processed_counter.setText(f"Обработано: {len(self.processed_paths)}")

        self.detection_counter.setText(f"Объектов: {data['total']}")

        self.avg_counter.setText(f"Среднее: {data['avg']}")
        self.time_counter.setText(f"Время: {data['time']}")

        self.total_detections = data["total"]

        self.gallery.display_results()

        self.status_text.setText("Обработанные результаты загружены")


    def toggle_input_images(self, checked):
        self.show_input_images = checked
        self.gallery.toggle_input_visibility(checked)


    def set_conf_threshold(self, value):
        self.detector.set_conf(value)
        self.status_text.setText(f"Порог обнаружения: {value}")


    def sort_results(self, mode):
        if not self.processed_paths:
            return

        combined = list(zip(
            self.processed_paths,
            self.result_pixmaps,
            self.per_image_detections
        ))

        if mode == "count":
            combined.sort(key=lambda x: x[2], reverse=True)
        else:
            combined.sort(key=lambda x: os.path.basename(x[0]))

        self.processed_paths, self.result_pixmaps, self.per_image_detections = zip(*combined)

        self.processed_paths = list(self.processed_paths)
        self.result_pixmaps = list(self.result_pixmaps)
        self.per_image_detections = list(self.per_image_detections)

        self.gallery.update_result_grid()
   

    def resizeEvent(self, event):
        if self.result_pixmaps:
            self.resize_timer.start(200)



    def select_image(self, index):
        self.open_viewer(index)

    def open_viewer(self, index):
        self.viewer = ImageViewer(self.images, index)
        self.viewer.show()

    def open_processed_viewer(self, index):
        self.proc_viewer = ImageViewer(self.processed_paths, index)
        self.proc_viewer.show()


    def detect_objects(self):

        self.clear_welcome_screen()

        if not self.images or self.processing:
            return

        self.results_saved = False
        self.processing = True
        self.detect_action.setEnabled(False)

        self.processing_start_time = time.time()

        self.processed_paths = []
        self.result_pixmaps = []
        self.per_image_detections = []
        self.total_detections = 0

        self.detection_counter.setText("Объектов: 0")
        self.avg_counter.setText("Среднее: 0.00")
        self.time_counter.setText("Время: 0.00 сек")

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.status_text.setText("Обработка изображений...")

        def progress(current, total, total_det):
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

            self.processed_counter.setText(f"Обработано: {current}")
            self.detection_counter.setText(f"Объектов: {total_det}")

            QApplication.processEvents()

        self.processed_paths, self.per_image_detections, self.total_detections = \
            self.detector.process_images(
                self.images,
                self.results_dir,
                progress_callback=progress,
                stop_flag=lambda: self.stop_requested
            )

        QApplication.restoreOverrideCursor()

        self.processing = False
        self.detect_action.setEnabled(True)

        self.processing_time = time.time() - self.processing_start_time

        avg = (
            self.total_detections / len(self.processed_paths)
            if self.processed_paths else 0
        )

        self.avg_counter.setText(f"Среднее: {avg:.2f}")
        self.time_counter.setText(f"Время: {self.processing_time:.2f} сек")

        QTimer.singleShot(0, self.gallery.display_results)
        self.status_text.setText("Обработка завершена")



    def closeEvent(self, event):
        #если идёт
        if self.loading_images or self.processing:

            if self.loading_images:
                text = "Сейчас выполняется загрузка изображений.\nВы уверены, что хотите выйти?"
            else:
                text = "Сейчас выполняется обработка изображений.\nВы уверены, что хотите выйти?"

            msg = QMessageBox(self)
            msg.setWindowTitle("Подтверждение выхода")
            msg.setText(text)

            yes_btn = msg.addButton("Выйти", QMessageBox.AcceptRole)
            no_btn = msg.addButton("Отмена", QMessageBox.RejectRole)

            msg.exec()

            if msg.clickedButton() == yes_btn:
                self.stop_requested = True
                event.accept()
            else:
                event.ignore()

            return

        #есть несохранённые
        if self.processed_paths and not self.results_saved:

            msg = QMessageBox(self)
            msg.setWindowTitle("Несохранённые результаты")
            msg.setText(
                "Результаты обработки не сохранены.\n"
                "Сохранить перед выходом?"
            )

            save_btn = msg.addButton("Сохранить", QMessageBox.AcceptRole)
            exit_btn = msg.addButton("Выйти без сохранения", QMessageBox.DestructiveRole)
            cancel_btn = msg.addButton("Отмена", QMessageBox.RejectRole)

            msg.exec()

            clicked = msg.clickedButton()

            if clicked == save_btn:
                self.save_results()

                #отмена
                if self.results_saved:
                    event.accept()
                else:
                    event.ignore()

            elif clicked == exit_btn:
                event.accept()

            else:
                event.ignore()

            return

        #обычный
        event.accept()