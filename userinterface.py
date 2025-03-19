import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QDockWidget,
    QAction, QToolBar, QSizePolicy
)
from PyQt5.QtCore import Qt
from agent import run_agent


class SearchUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Catechism Search")
        self.resize(800, 600)
        
        self.previous_questions = []
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Conversation display (read-only)
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.main_layout.addWidget(self.conversation_display)
        
        # Input area: QLineEdit and Send button
        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_question)
        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.send_button)
        self.main_layout.addLayout(self.input_layout)
        
        # Left dock for previous questions (initially hidden)
        self.left_dock = QDockWidget("Previous Questions", self)
        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.question_list = QListWidget()
        self.left_dock.setWidget(self.question_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.left_dock.hide()
        
        # Toolbar with menu toggle (hamburger) and new page (pencil)
        self.setup_toolbar()

    def setup_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        # Menu toggle action (hamburger icon)
        self.menu_action = QAction("☰", self)
        self.menu_action.triggered.connect(self.toggle_left_menu)
        toolbar.addAction(self.menu_action)
        
        # Spacer widget to push the pencil icon to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Pencil icon action for new page
        self.new_page_action = QAction("✏", self)
        self.new_page_action.triggered.connect(self.new_page)
        toolbar.addAction(self.new_page_action)

    def toggle_left_menu(self):
        # Show or hide the left dock with previous questions
        if self.left_dock.isVisible():
            self.left_dock.hide()
        else:
            self.left_dock.show()
            
    def new_page(self):
        # Clear conversation display and previous questions
        self.conversation_display.clear()
        self.previous_questions = []
        self.question_list.clear()
        
    def send_question(self):
        # Retrieve text from input field
        question = self.input_field.text().strip()
        if not question:
            return
        
        # Append the user's question to the conversation display
        self.conversation_display.append(f"You: {question}")
        
        # Store and display the question in the previous questions list
        self.previous_questions.append(question)
        self.question_list.addItem(question)
        
        # Get the answer using your function and display it
        answer = run_agent(question)
        self.conversation_display.append(f"Answer: {answer}\n")
        
        # Clear the input field for the next question
        self.input_field.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchUI()
    window.show()
    sys.exit(app.exec_())

