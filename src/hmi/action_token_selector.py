from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QLabel, QComboBox,
    QSpacerItem, QSizePolicy, QVBoxLayout
)
import sys
from collections import deque

action_tokens = ['Delay', 'Stop', 'Prioritise']


class ActionTokenSelector(QWidget):
    def __init__(self, trainIDs = None, parent = None):
        super().__init__(parent)
        if trainIDs:
            self.trainIDs = [str(trainID) for trainID in trainIDs]
        else:
            self.trainIDs = []

        # Main horizontal layout for the whole row
        self.mainlayout = QHBoxLayout(self)
        self.setLayout(self.mainlayout)

        # Action Tokens (label + combo)
        action_layout = QVBoxLayout()
        self.action_label = QLabel("Action Token:")
        self.action_combo = QComboBox()
        self.action_combo.addItems(['Select Action Token'] + action_tokens)
        action_layout.addWidget(self.action_label)
        action_layout.addWidget(self.action_combo)
        action_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        action_container = QWidget()
        action_container.setLayout(action_layout)
        self.mainlayout.addWidget(action_container)

        # Dynamic Widget Placeholder (horizontal layout)
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QHBoxLayout()
        self.dynamic_widget.setLayout(self.dynamic_layout)
        self.mainlayout.addWidget(self.dynamic_widget)
        self.mainlayout.addStretch(1)

        self.action_combo.currentIndexChanged.connect(self._update_dynamic_widget)


    def _update_dynamic_widget(self, action_token):
        action_token = self.action_combo.currentText()
        print(f"Selected Action Token: {action_token}")
        while self.dynamic_layout.count():
            child = self.dynamic_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        def add_label_dropdown(label_text, trainIDs):
            vbox = QVBoxLayout()
            label = QLabel(label_text)
            dropdown = self._create_train_dropdown(trainIDs)
            vbox.addWidget(label)
            vbox.addWidget(dropdown)
            container = QWidget()
            container.setLayout(vbox)
            self.dynamic_layout.addWidget(container)

        if action_token in action_tokens:
            if action_token == 'Delay':
                add_label_dropdown("Train to Delay:", self.trainIDs)
            elif action_token == 'Stop':
                add_label_dropdown("Train to Stop:", self.trainIDs)
            elif action_token == 'Prioritise':
                add_label_dropdown("Primary Train to Prioritise:", self.trainIDs)
                add_label_dropdown("Secondary Train to Prioritise:", self.trainIDs)

        self.dynamic_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.dynamic_widget.setLayout(self.dynamic_layout)


    def _create_train_dropdown(self, trainIDs):
        train_dropdown = QComboBox()
        train_dropdown.addItems(trainIDs)
        return train_dropdown
    

    def clear_tokens(self):
        """Clear all action tokens."""
        self.action_combo.clear()
        self.action_combo.addItems(['Select Action Token'] + action_tokens)
        self._update_dynamic_widget(0)


    def get_tokens(self):
        """Get the currently selected action token and its associated data."""
        action_token = self.action_combo.currentText()
        if action_token == 'Select Action Token':
            return None
        
        token_data = {0: action_token}
        for i in range(self.dynamic_layout.count()):
            container = self.dynamic_layout.itemAt(i).widget()
            if container:
                dropdown = container.findChild(QComboBox)
                if dropdown:
                    token_data[i+1] = dropdown.currentText()
        
        return token_data
