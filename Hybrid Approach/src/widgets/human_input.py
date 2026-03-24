from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QSizePolicy, QHBoxLayout, QFrame, QStyle, QDialog, QComboBox, QDialogButtonBox
from PyQt6.QtCore import Qt, pyqtSignal
from src.widgets.action_token_selector import ActionTokenSelector
from src.utils.env_reference import FlatlandEnvReference

class HumanInputWidget(QFrame):
    # Signals
    tokens_signal = pyqtSignal(dict)

    def __init__(self, env_ref: FlatlandEnvReference, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        tokens_layout = QHBoxLayout(self)
        
        # Apply Button
        apply_btn = QPushButton()
        apply_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        apply_btn.setFixedSize(32, 32)
        apply_btn.clicked.connect(lambda: self.send_signal_with_tokens())
        self.token_selector = ActionTokenSelector(env_ref.env.get_agent_handles())
        delete_btn = QPushButton()
        delete_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
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
        tokens: dict = self.token_selector.get_tokens()
        self.tokens_signal.emit(tokens)