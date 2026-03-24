from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStyle,
)
from PyQt6.QtCore import pyqtSignal

from src.hmi.action_token_selector import ActionTokenSelector


class HumanInputWidget(QFrame):
    # Signal with selected tokens
    tokens_signal = pyqtSignal(dict)

    def __init__(self, agent_handles=None, parent=None):
        super().__init__(parent)

        if agent_handles is None:
            agent_handles = []

        self.setFrameShape(QFrame.Shape.Box)
        tokens_layout = QHBoxLayout(self)

        # Apply button
        apply_btn = QPushButton()
        apply_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        apply_btn.setFixedSize(32, 32)
        apply_btn.clicked.connect(self.send_signal_with_tokens)

        # Token selector
        self.token_selector = ActionTokenSelector(agent_handles)

        # Delete button
        delete_btn = QPushButton()
        delete_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        delete_btn.setFixedSize(32, 32)
        delete_btn.clicked.connect(self.on_delete_clicked)

        tokens_layout.addWidget(apply_btn)
        tokens_layout.addWidget(self.token_selector, stretch=1)
        tokens_layout.addWidget(delete_btn)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(256)

    def on_delete_clicked(self):
        self.token_selector.clear_tokens()

    def send_signal_with_tokens(self):
        """Send a signal with the values of the ActionTokenSelector."""
        tokens = self.token_selector.get_tokens()
        self.tokens_signal.emit(tokens)