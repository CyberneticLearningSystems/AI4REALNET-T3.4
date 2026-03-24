"""
Human-in-the-Loop System for Compact Junction Environment
==========================================================
GUI-based system that:
1. Loads pretrained model
2. Runs episodes with model predictions
3. Pauses at critical decisions (conflict + junction)
4. Shows model recommendation
5. Human clicks to accept/modify
6. Visual warnings for all conflicts (non-blocking)


"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QMessageBox, QGroupBox, QRadioButton, QButtonGroup)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from stable_baselines3 import PPO

# Import local modules
from compact_junction_env import CompactJunctionEnv
from conflict_detector_compact import ConflictDetector
from corridor_visualization_widget import CorridorVisualizationWidget


class HumanInTheLoopSystem(QMainWindow):
    """GUI-based human-in-the-loop system"""
    
    def __init__(self, models=None, train_mode=False, learning_rate=1e-5):
        super().__init__()
        
        self.models = models or {}  # Dictionary of {name: path}
        self.loaded_models = {}     # Dictionary of {name: PPO_model}
        self.train_mode = train_mode
        self.learning_rate = learning_rate
        
        # State
        self.env = None
        self.conflict_detector = None
        self.paused_for_decision = False
        self.pending_recommendations = None
        self.step_count = 0
        self.episode_count = 0
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'human_choices': {},  
            'model_agreements': 0,
            'model_disagreements': 0,
            'conflicts_avoided': 0,
            'collisions': 0,
            'success_count': 0,
            'episode_rewards': []
        }
        
        # Storage for training (if in train mode)
        self.human_corrections = []  # (obs, action) pairs
        
        # Setup UI
        self.setup_ui()
        
        # Load environment and models
        self._load_environment()
        self._load_models()
        
        # Timer for auto-stepping
        self.timer = QTimer()
        self.timer.timeout.connect(self._auto_step)
        self.step_delay = 500  # ms
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Human-in-the-Loop System - Compact Junction")
        self.resize(1600, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left side: Visualization
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.viz_widget = CorridorVisualizationWidget()
        left_layout.addWidget(self.viz_widget)
        
        # Warning display
        self.warning_label = QLabel("No conflicts detected")
        self.warning_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #4CAF50;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.warning_label.setWordWrap(True)
        left_layout.addWidget(self.warning_label)
        
        main_layout.addWidget(left_widget, stretch=2)
        
        # Right side: Controls and info
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Episode controls
        controls_group = QGroupBox("Episode Controls")
        controls_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Episode")
        self.start_btn.clicked.connect(self.start_episode)
        controls_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_episode)
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_episode)
        controls_layout.addWidget(self.reset_btn)
        
        # Train mode toggle
        self.train_mode_toggle = QPushButton("🎓 Enable Training Mode")
        self.train_mode_toggle.setCheckable(True)
        self.train_mode_toggle.setChecked(self.train_mode)
        self.train_mode_toggle.clicked.connect(self._toggle_train_mode)
        if self.train_mode:
            self.train_mode_toggle.setText("🎓 Training Mode: ON")
            self.train_mode_toggle.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.train_mode_toggle.setText("🎓 Training Mode: OFF")
            self.train_mode_toggle.setStyleSheet("background-color: #9E9E9E; color: white;")
        controls_layout.addWidget(self.train_mode_toggle)
        
        # Conflict visualization toggle
        self.show_conflicts_btn = QPushButton("⚠️ Show Conflicts")
        self.show_conflicts_btn.setCheckable(True)
        self.show_conflicts_btn.setChecked(False)
        self.show_conflicts_btn.clicked.connect(self._toggle_conflicts)
        self.show_conflicts_btn.setStyleSheet("background-color: #FF9800; color: white;")
        controls_layout.addWidget(self.show_conflicts_btn)
        
        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)
        
        # Decision panel (shows when critical decision needed)
        self.decision_group = QGroupBox("⚠️ CRITICAL DECISION REQUIRED")
        decision_layout = QVBoxLayout()
        
        self.situation_text = QTextEdit()
        self.situation_text.setReadOnly(True)
        self.situation_text.setMaximumHeight(300)
        self.situation_text.setFont(QFont("Courier", 9))
        decision_layout.addWidget(self.situation_text)
        
        # Model choice section
        choice_label = QLabel("Choose Model Recommendation:")
        choice_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        decision_layout.addWidget(choice_label)
        
        # Container for dynamic radio buttons
        self.radio_button_container = QWidget()
        self.radio_button_layout = QVBoxLayout()
        self.radio_button_container.setLayout(self.radio_button_layout)
        decision_layout.addWidget(self.radio_button_container)
        
        # Radio button group (will be populated dynamically)
        self.model_button_group = QButtonGroup()
        
        # Action choice buttons
        self.accept_btn = QPushButton("✓ Accept Selected Recommendation")
        self.accept_btn.clicked.connect(self.accept_recommendation)
        self.accept_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        decision_layout.addWidget(self.accept_btn)
        
        # Model agreement indicator
        self.agreement_label = QLabel("")
        self.agreement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.agreement_label.setWordWrap(True)
        decision_layout.addWidget(self.agreement_label)
        
        self.decision_group.setLayout(decision_layout)
        self.decision_group.setVisible(False)  # Hidden until needed
        right_layout.addWidget(self.decision_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(250)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Info display
        self.info_label = QLabel("Load a model to begin")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        right_layout.addWidget(self.info_label)
        
        right_layout.addStretch()
        
        main_layout.addWidget(right_widget, stretch=1)
        
        self._update_stats_display()
    
    def _load_environment(self):
        """Load the compact junction environment"""
        try:
            self.env = CompactJunctionEnv()
            self.conflict_detector = ConflictDetector(self.env)
            self.viz_widget.set_environment(self.env)
            self.info_label.setText("✓ Environment loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load environment:\n{e}")
    
    def _load_models(self):
        """Load all models from the models dictionary"""
        self.loaded_models = {}
        models_loaded = []
        
        for name, path in self.models.items():
            try:
                model_path = Path(path)
                if not model_path.exists() and model_path.with_suffix('.zip').exists():
                    model_path = model_path.with_suffix('.zip')
                
                self.loaded_models[name] = PPO.load(str(model_path))
                models_loaded.append(name)
                
                # Initialize stats counter for this model
                self.stats['human_choices'][name] = 0
                
            except Exception as e:
                print(f"Warning: Failed to load {name} model from {path}: {e}")
        
        # Update info label
        if models_loaded:
            mode_str = "TRAIN" if self.train_mode else "PREDICT"
            self.info_label.setText(
                f"✓ Models loaded: {', '.join(models_loaded)}\n"
                f"Mode: {mode_str}\n"
                f"{'Learning rate: ' + str(self.learning_rate) if self.train_mode else ''}"
            )
        else:
            self.info_label.setText("⚠️ No models loaded - will use random actions")
            QMessageBox.warning(self, "Warning", 
                              "No models loaded! Provide at least one model path.\n"
                              "System will use random actions.")
    
    def _toggle_train_mode(self):
        """Toggle between training and prediction mode"""
        self.train_mode = not self.train_mode
        
        if self.train_mode:
            self.train_mode_toggle.setText("🎓 Training Mode: ON")
            self.train_mode_toggle.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.info_label.setText(
                self.info_label.text().replace("Mode: PREDICT", "Mode: TRAIN")
            )
        else:
            self.train_mode_toggle.setText("🎓 Training Mode: OFF")
            self.train_mode_toggle.setStyleSheet("background-color: #9E9E9E; color: white;")
            self.info_label.setText(
                self.info_label.text().replace("Mode: TRAIN", "Mode: PREDICT")
            )
    
    def _toggle_conflicts(self):
        """Toggle conflict visualization on/off"""
        show = self.show_conflicts_btn.isChecked()
        
        if show:
            self.show_conflicts_btn.setText("⚠️ Hide Conflicts")
            self.show_conflicts_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
            
            print("DEBUG: Show Conflicts button clicked - ON")
            
            # Detect conflicts and get positions WHERE they will happen
            conflict_positions = []
            
            if hasattr(self, 'conflict_detector'):
                is_critical, conflicts = self.conflict_detector.is_critical_decision_point()
                print(f"DEBUG: is_critical = {is_critical}, num_conflicts = {len(conflicts)}")
                
                # Extract positions from conflicts
                for conflict in conflicts:
                    # Debug: print conflict to see what's in it
                    print(f"DEBUG Conflict: {conflict}")
                    
                    # Check for 'positions' field (plural) - it's a list!
                    if 'positions' in conflict:
                        for pos in conflict['positions']:
                            conflict_positions.append(pos)
                            print(f"DEBUG: Added position from 'positions' list: {pos}")
                    
                    # Check for 'position' field (singular)
                    elif 'position' in conflict:
                        conflict_positions.append(conflict['position'])
                        print(f"DEBUG: Using position from conflict: {conflict['position']}")
                    
                    # Otherwise try to get it from type-specific info
                    elif 'type' in conflict:
                        if conflict['type'] == 'junction' and 'junction_pos' in conflict:
                            conflict_positions.append(conflict['junction_pos'])
                            print(f"DEBUG: Using junction_pos: {conflict['junction_pos']}")
                        elif conflict['type'] == 'head_on' and 'collision_pos' in conflict:
                            conflict_positions.append(conflict['collision_pos'])
                            print(f"DEBUG: Using collision_pos: {conflict['collision_pos']}")
                
                # Remove duplicates
                conflict_positions = list(set(conflict_positions))
                print(f"DEBUG: Final conflict_positions: {conflict_positions}")
            
            # Update visualization
            print(f"DEBUG: Calling toggle_conflicts(True) and set_conflict_positions({conflict_positions})")
            self.viz_widget.set_conflict_positions(conflict_positions)
            self.viz_widget.toggle_conflicts(True)
        else:
            print("DEBUG: Show Conflicts button clicked - OFF")
            self.show_conflicts_btn.setText("⚠️ Show Conflicts")
            self.show_conflicts_btn.setStyleSheet("background-color: #FF9800; color: white;")
            self.viz_widget.set_conflict_positions([])
            self.viz_widget.toggle_conflicts(False)
    
    def _populate_radio_buttons(self):
        """Dynamically create radio buttons for loaded models"""
        # Clear existing radio buttons
        for i in reversed(range(self.radio_button_layout.count())):
            self.radio_button_layout.itemAt(i).widget().setParent(None)
        
        # Clear button group
        for button in self.model_button_group.buttons():
            self.model_button_group.removeButton(button)
        
        # Create radio button for each loaded model
        for i, name in enumerate(self.loaded_models.keys()):
            radio = QRadioButton(name)
            radio.setStyleSheet("QRadioButton { padding: 5px; }")
            self.model_button_group.addButton(radio, i)
            self.radio_button_layout.addWidget(radio)
            
            # Check first button by default
            if i == 0:
                radio.setChecked(True)
    
    def start_episode(self):
        """Start a new episode"""
        if self.env is None:
            QMessageBox.warning(self, "Warning", "Environment not loaded!")
            return
        
        # Reset environment
        self.env.reset()
        self.conflict_detector.last_moves = {}
        self.step_count = 0
        self.episode_count += 1
        self.paused_for_decision = False
        self.pending_recommendation = None
        
        # Update display
        self.viz_widget.update_visualization()
        self._check_conflicts()
        
        # Enable/disable buttons
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        
        # Start auto-stepping
        self.timer.start(self.step_delay)
        
        self.info_label.setText(f"Episode {self.episode_count} started...")
    
    def pause_episode(self):
        """Pause the current episode"""
        self.timer.stop()
        self.pause_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ Resume")
        self.info_label.setText("⏸ Episode paused")
    
    def reset_episode(self):
        """Reset the current episode"""
        self.timer.stop()
        if self.env:
            self.env.reset()
            self.viz_widget.update_visualization()
        
        self.paused_for_decision = False
        self.pending_recommendation = None
        self.decision_group.setVisible(False)
        
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ Start Episode")
        self.pause_btn.setEnabled(False)
        
        self.info_label.setText("Episode reset")
        self._check_conflicts()
    
    def _auto_step(self):
        """Execute one step automatically"""
        if self.paused_for_decision:
            # Waiting for human input
            return
        
        # Check if episode is done
        if all(self.env.agents_done) or self.env.step_count >= self.env.max_steps:
            self._episode_complete()
            return
        
        # Check for critical decision
        is_critical, conflicts = self.conflict_detector.is_critical_decision_point()
        
        if is_critical:
            # PAUSE and ask human
            self._request_human_decision(conflicts)
            return
        
        # Normal step: use model prediction
        self._execute_model_action()
    
    def _execute_model_action(self):
        """Execute action from model prediction (uses first available model)"""
        obs = self.env._get_observation()
        
        # Use first available model
        if self.loaded_models:
            first_model = next(iter(self.loaded_models.values()))
            action, _ = first_model.predict(obs, deterministic=True)
        else:
            # Random fallback
            action = self.env.action_space.sample()
        
        # Execute action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update conflict detector
        for agent_id in range(self.env.n_agents):
            self.conflict_detector.update_last_move(agent_id, action[agent_id])
        
        # Update display
        self.step_count += 1
        self.viz_widget.update_visualization()
        self._check_conflicts()
        
        # Check if episode done
        if all(self.env.agents_done) or self.env.step_count >= self.env.max_steps:
            self._episode_complete()
    
    def _request_human_decision(self, conflicts):
        """Pause and request human decision"""
        self.paused_for_decision = True
        self.timer.stop()
        self.stats['total_decisions'] += 1
        
        # Get model recommendations from all loaded models
        obs = self.env._get_observation()
        
        recommendations = {}
        for name, model in self.loaded_models.items():
            action, _ = model.predict(obs, deterministic=True)
            recommendations[name] = action
        
        # Check if all models agree
        all_agree = False
        if len(recommendations) > 1:
            actions = list(recommendations.values())
            all_agree = all(np.array_equal(actions[0], a) for a in actions[1:])
            
            if all_agree:
                self.stats['model_agreements'] += 1
            else:
                self.stats['model_disagreements'] += 1
        
        recommendations['all_agree'] = all_agree
        
        self.pending_recommendations = (obs, recommendations)
        
        # Populate radio buttons for this decision
        self._populate_radio_buttons()
        
        # Display situation
        self._display_critical_decision(conflicts, recommendations)
        
        # Update conflict visualization if enabled - show WHERE conflicts will happen
        if self.show_conflicts_btn.isChecked():
            conflict_positions = []
            
            # Extract actual conflict positions from conflict data
            for conflict in conflicts:
                # Check for 'positions' field (plural) - it's a list!
                if 'positions' in conflict:
                    for pos in conflict['positions']:
                        conflict_positions.append(pos)
                
                # Check for 'position' field (singular)
                elif 'position' in conflict:
                    conflict_positions.append(conflict['position'])
                
                # Otherwise try type-specific position fields  
                elif 'type' in conflict:
                    if conflict['type'] == 'junction' and 'junction_pos' in conflict:
                        conflict_positions.append(conflict['junction_pos'])
                    elif conflict['type'] == 'head_on' and 'collision_pos' in conflict:
                        conflict_positions.append(conflict['collision_pos'])
            
            # Remove duplicates
            conflict_positions = list(set(conflict_positions))
            
            if conflict_positions:
                self.viz_widget.set_conflict_positions(conflict_positions)
        
        # Show decision panel
        self.decision_group.setVisible(True)
    
    def _display_critical_decision(self, conflicts, recommendations):
        """Display the critical decision information with all model recommendations"""
        action_names = {0: "North", 1: "East", 2: "South", 3: "West", 4: "WAIT"}
        colors = ['Red', 'Cyan', 'Yellow']
        
        # Build situation text with clean formatting
        situation = "=" * 60 + "\n"
        situation += "  🚨 CRITICAL DECISION POINT - Human Input Required\n"
        situation += "=" * 60 + "\n\n"
        situation += "SITUATION:\n"
        situation += "-" * 60 + "\n"
        
        # Show agent positions and junction status
        for i in range(self.env.n_agents):
            if not self.env.agents_done[i]:
                pos = self.env.agents_pos[i]
                at_junc = "✓ AT JUNCTION" if self.conflict_detector.is_at_junction(i) else ""
                situation += f"  {colors[i]} Agent {i}: Position ({pos[0]},{pos[1]}) {at_junc}\n"
        
        situation += "\nDETECTED CONFLICTS:\n"
        situation += "-" * 60 + "\n"
        
        for conflict in conflicts:
            desc = self.conflict_detector.get_conflict_description(conflict)
            situation += f"  • {desc}\n"
        
        situation += "\n" + "=" * 60 + "\n"
        situation += "MODEL RECOMMENDATIONS:\n"
        situation += "=" * 60 + "\n"
        
        # Show each model's recommendation
        model_num = 1
        for name, action in recommendations.items():
            if name == 'all_agree':
                continue  # Skip the metadata field
            
            situation += f"\n[{model_num}] {name.upper()}\n"
            situation += "-" * 60 + "\n"
            for i in range(self.env.n_agents):
                action_name = action_names.get(action[i], "?")
                situation += f"    {colors[i]} Agent {i}: {action_name}\n"
            model_num += 1
        
        situation += "\n" + "=" * 60
        
        self.situation_text.setPlainText(situation)
        
        # Update agreement indicator
        if recommendations.get('all_agree', False):
            self.agreement_label.setText("✓ All models AGREE on this recommendation")
            self.agreement_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
        elif len(self.loaded_models) > 1:
            self.agreement_label.setText("⚠ Models DISAGREE - choose carefully!")
            self.agreement_label.setStyleSheet("color: #ff9800; font-weight: bold; padding: 5px;")
        else:
            self.agreement_label.setText("")
    
    def accept_recommendation(self):
        """Human accepts one of the model recommendations based on radio button selection"""
        if not self.pending_recommendations:
            return
        
        obs, recommendations = self.pending_recommendations
        
        # Get selected radio button
        selected_button = self.model_button_group.checkedButton()
        if not selected_button:
            QMessageBox.warning(self, "Error", "No model selected!")
            return
        
        # Get model name from selected button
        selected_model_name = selected_button.text()
        
        # Get corresponding action
        if selected_model_name not in recommendations or selected_model_name == 'all_agree':
            QMessageBox.warning(self, "Error", "Invalid model selection!")
            return
        
        action = recommendations[selected_model_name]
        
        # Record choice
        if selected_model_name not in self.stats['human_choices']:
            self.stats['human_choices'][selected_model_name] = 0
        self.stats['human_choices'][selected_model_name] += 1
        
        # Store for training if in train mode
        if self.train_mode:
            self.human_corrections.append((obs.copy(), action.copy()))
        
        # Execute the action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update conflict detector
        for agent_id in range(self.env.n_agents):
            self.conflict_detector.update_last_move(agent_id, action[agent_id])
        
        # Resume
        self.paused_for_decision = False
        self.pending_recommendations = None
        self.decision_group.setVisible(False)
        
        # Update display
        self.step_count += 1
        self.viz_widget.update_visualization()
        self._check_conflicts()
        self._update_stats_display()
        
        # Resume auto-stepping
        if not all(self.env.agents_done) and self.env.step_count < self.env.max_steps:
            self.timer.start(self.step_delay)
        else:
            self._episode_complete()
    
    def _check_conflicts(self):
        """Check for conflicts and update warning display"""
        warnings = self.conflict_detector.get_all_warnings()
        
        if warnings:
            warning_text = "\n".join(warnings)
            self.warning_label.setText(warning_text)
            self.warning_label.setStyleSheet("""
                QLabel {
                    background-color: #ff9800;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
        else:
            self.warning_label.setText("✓ No conflicts detected")
            self.warning_label.setStyleSheet("""
                QLabel {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
        
        # Update conflict visualization if enabled - show WHERE conflicts will happen
        if self.show_conflicts_btn.isChecked():
            is_critical, conflicts = self.conflict_detector.is_critical_decision_point()
            conflict_positions = []
            
            # Extract actual conflict positions from conflict data
            for conflict in conflicts:
                # Check for 'positions' field (plural) - it's a list!
                if 'positions' in conflict:
                    for pos in conflict['positions']:
                        conflict_positions.append(pos)
                
                # Check for 'position' field (singular)
                elif 'position' in conflict:
                    conflict_positions.append(conflict['position'])
                
                # Otherwise try type-specific position fields
                elif 'type' in conflict:
                    if conflict['type'] == 'junction' and 'junction_pos' in conflict:
                        conflict_positions.append(conflict['junction_pos'])
                    elif conflict['type'] == 'head_on' and 'collision_pos' in conflict:
                        conflict_positions.append(conflict['collision_pos'])
            
            # Remove duplicates
            conflict_positions = list(set(conflict_positions))
            self.viz_widget.set_conflict_positions(conflict_positions)
    
    def _episode_complete(self):
        """Handle episode completion"""
        self.timer.stop()
        
        # Calculate statistics
        success = all(self.env.agents_done)
        
        if success:
            self.stats['success_count'] += 1
        
        # Show completion message
        result = "SUCCESS! ✓" if success else "INCOMPLETE"
        msg = f"Episode {self.episode_count} complete: {result}\n"
        msg += f"Steps taken: {self.env.step_count}\n"
        
        if success:
            msg += f"All agents reached goals!"
        else:
            completed = sum(self.env.agents_done)
            msg += f"Agents completed: {completed}/{self.env.n_agents}"
        
        self.info_label.setText(msg)
        
        # Update stats
        self._update_stats_display()
        
        # Re-enable start button
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ Start New Episode")
        self.pause_btn.setEnabled(False)
    
    def _update_stats_display(self):
        """Update the statistics display"""
        total_choices = sum(self.stats['human_choices'].values())
        
        # Build model choices display
        model_choices_str = ""
        for model_name, count in self.stats['human_choices'].items():
            pct = 100 * count / max(1, total_choices)
            model_choices_str += f"  {model_name:20s} {count} ({pct:.1f}%)\n"
        
        stats_text = f"""
OVERALL STATISTICS
{'='*40}

Episodes Completed: {self.episode_count}
Success Rate: {self.stats['success_count']}/{self.episode_count} ({100*self.stats['success_count']/max(1,self.episode_count):.1f}%)

HUMAN DECISIONS
{'='*40}

Critical Decisions: {self.stats['total_decisions']}

Model Choices:
{model_choices_str}
MODEL AGREEMENT
{'='*40}

Agreements:    {self.stats['model_agreements']}
Disagreements: {self.stats['model_disagreements']}

TRAINING DATA
{'='*40}

Corrections Stored: {len(self.human_corrections)}
Mode: {'TRAIN' if self.train_mode else 'PREDICT'}
"""
        
        self.stats_text.setPlainText(stats_text)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Human-in-the-Loop System - Flexible Model Loading')
    
    # Predefined model slots
    parser.add_argument('--safe', type=str, default=None,
                       help='Path to safe model')
    parser.add_argument('--balanced', type=str, default=None,
                       help='Path to balanced model')
    parser.add_argument('--efficient', type=str, default=None,
                       help='Path to efficient model')
    
    # Generic model loading (can add as many as you want!)
    parser.add_argument('--model', type=str, action='append',
                       help='Generic model: "path" or "name:path" (can use multiple times)')
    
    # Training mode
    parser.add_argument('--train', action='store_true',
                       help='Enable training mode (collect human corrections)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for training mode')
    
    args = parser.parse_args()
    
    # Build model dictionary
    models = {}
    
    # Add predefined models
    if args.safe:
        models['Safe 🛡️'] = args.safe
    if args.balanced:
        models['Balanced ⚖️'] = args.balanced
    if args.efficient:
        models['Efficient ⚡'] = args.efficient
    
    # Add generic models
    if args.model:
        for i, model_spec in enumerate(args.model):
            if ':' in model_spec:
                # Format: "name:path"
                name, path = model_spec.split(':', 1)
                models[name] = path
            else:
                # Just path - auto-generate name
                models[f'Model {i+1}'] = model_spec
    
    # Check if at least one model is provided
    if not models:
        print("Warning: No models specified. System will use random actions.")
        print()
        print("Usage Examples:")
        print("  # Single model (generic):")
        print("  python human_in_loop_compact.py --model models/my_model.zip")
        print()
        print("  # Two models with custom names:")
        print("  python human_in_loop_compact.py \\")
        print("    --model 'Fast:models/fast.zip' \\")
        print("    --model 'Safe:models/safe.zip'")
        print()
        print("  # Three predefined models:")
        print("  python human_in_loop_compact.py \\")
        print("    --safe models/safe_final.zip \\")
        print("    --balanced models/balanced_final.zip \\")
        print("    --efficient models/efficient_final.zip")
        print()
        print("  # Mix and match:")
        print("  python human_in_loop_compact.py \\")
        print("    --balanced models/balanced_final.zip \\")
        print("    --model 'Custom:models/custom.zip'")
        print()
    
    app = QApplication(sys.argv)
    
    window = HumanInTheLoopSystem(
        models=models,
        train_mode=args.train,
        learning_rate=args.lr
    )
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()