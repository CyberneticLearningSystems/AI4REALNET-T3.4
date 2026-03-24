import sys
from pathlib import Path

from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from flatland.envs.observations import GlobalObsForRailEnv

from run_controller import HybridController, NegotiationManager, Token
from src.environments.scenario_loader import load_scenario_from_json
from src.utils.env_reference import FlatlandEnvReference
from src.widgets.human_input import HumanInputWidget


class HMIDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HMI Demo")
        self.resize(700, 300)
        # to run: python app_hmi_demo.py
        scenario_path = Path("src/environments/simple_avoidance.json")                
        # scenario_path = Path("src/environments/simple_avoidance_1.json")                      
        # scenario_path = Path("src/environments/simple_avoidance_2.json")              
        # scenario_path = Path("src/environments/simple_ordering.json")                 
        # scenario_path = Path("src/environments/simple_ordering_for_vis.json")          
        # scenario_path = Path("src/environments/overtaking.json")                      
        # scenario_path = Path("src/environments/network_experiments.json")            
        # scenario_path = Path("src/environments/graph_test_connected_schedules.json")  
        # scenario_path = Path("src/environments/complex_avoidance.json")               
        # scenario_path = Path("src/environments/complex_avoidance_1.json")             
        self.env = load_scenario_from_json(
            str(scenario_path),
            observation_builder=GlobalObsForRailEnv(),
            max_agents=None,
        )
        self.env.reset()

        self.renderer = RenderTool(
            self.env,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False,
            screen_height=600,
            screen_width=800,
        )

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        self.env_ref = FlatlandEnvReference(self.env)

        self.negotiator = NegotiationManager()
        self.controller = HybridController(mode="cbs")
        self._last_token_text = "none"
        self._priority_agent = None  

        self.status_label = QLabel("Ready, simulation paused.")
        self.info_label = QLabel("No token selected.")

        self.human_input = HumanInputWidget(self.env_ref)
        self.human_input.tokens_signal.connect(self.on_tokens_received)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)

        self.stop_button = QPushButton("Pause")
        self.stop_button.clicked.connect(self.stop_simulation)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.human_input)
        layout.addWidget(self.canvas)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)

        self.render_env()
        self.print_scenario_summary(scenario_path) 


    def print_scenario_summary(self, scenario_path):
        print(f"\n Scenario loaded: {scenario_path.name}")
        print(f"Agents: {self.env.number_of_agents} | size: {self.env.width} x {self.env.height}")

        for i, agent in enumerate(self.env.agents):
            print(f"agent{i} start: {agent.initial_position}")
            print(f"agent{i} target: {agent.target}")
            print(f"agent{i} earliest_departure: {agent.earliest_departure}")

    def print_step_summary(self, rewards, dones):
        token_count = len(getattr(self, "_last_tokens", []))
        print(
            f"\n=== t={self.env._elapsed_steps} mode={self.controller.mode} "
            f"tokens={token_count} last_token={self._last_token_text} ==="
        )

        for i, agent in enumerate(self.env.agents):
            reward = rewards.get(i, 0) if isinstance(rewards, dict) else 0
            done = dones.get(i, False) if isinstance(dones, dict) else False
            print(
                f"agent{i}: pos={agent.position} state={agent.state} reward={reward} done={done}"
            )


    def on_tokens_received(self, token_dict):
        if not token_dict:
            self.info_label.setText("No token selected.")
            return

        action = token_dict.get(0)

        if action == "Prioritise":
            primary = token_dict.get(1)
            secondary = token_dict.get(2)

            if primary is not None and primary != "":
                primary = int(primary)

                if secondary is not None and secondary != "":
                    secondary = int(secondary)
                else:
                    secondary = None

                self.negotiator.add(
                    Token(kind="PRIORITY", agent=primary, payload={"boost": 10})
                )

                self._priority_agent = primary

                self.info_label.setText(f"PRIORITY Token for agent {primary} generated.")

                print(f"\nPRIORITY token applied: primary_agent={primary}, secondary_agent={secondary}")

        elif action == "Stop":
            agent = token_dict.get(1)
            if agent is not None and agent != "":
                agent = int(agent)
                self.negotiator.add(
                    Token(kind="STOP", agent=agent, payload={})
                )
                self.info_label.setText(f"STOP Token for agent {agent} generated.")

        elif action == "Delay":
            agent = token_dict.get(1)
            if agent is not None and agent !="":
                agent = int(agent)
                self.negotiator.add(
                    Token(kind="DELAY", agent=agent, payload={})
                )
                self.info_label.setText(f"DELAY Token for agent {agent} generated.")

        else:
            self.info_label.setText(f"Unknown token: {action}")

    def start_simulation(self):
        self.timer.start(800)
        self.status_label.setText("Simulation running ...")

    def stop_simulation(self):
        self.timer.stop()
        self.status_label.setText("Simulation paused.")

    def _get_agent_display_position(self, agent):
        if agent.position is not None:
            return agent.position

        if hasattr(agent, "state") and agent.state == 6 and agent.target is not None:
            return agent.target

        if hasattr(agent, "initial_position") and agent.initial_position is not None:
            return agent.initial_position

        return None

    
    def render_env(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        self.renderer.render_env(
            show=False,
            frames=False,
            show_observations=False,
            show_predictions=False,
        )

        image = self.renderer.get_image()
        ax.imshow(image)
        ax.axis("off")

        img_h, img_w = image.shape[0], image.shape[1]

        for i, agent in enumerate(self.env.agents):
            pos = self._get_agent_display_position(agent)
            if pos is None:
                continue

            row, col = pos

            x = (col + 0.5) * img_w / self.env.width
            y = (row + 0.5) * img_h / self.env.height

            
            label = f"★{i}" if i == self._priority_agent else str(i)  

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
            )

        self.canvas.draw()

    def step_simulation(self):
        tokens = self.negotiator.consume_all()
        self._last_tokens = tokens

        actions = self.controller.act(self.env, self.env._elapsed_steps, tokens)
        
        obs, rewards, dones, infos = self.env.step(actions)
        self.render_env()
        self.print_step_summary(rewards, dones)

        agent_states = []
        for i, agent in enumerate(self.env.agents):
            agent_states.append(f"A{i}: pos={agent.position}, state={agent.state}")

        self.status_label.setText(
            f"t={self.env._elapsed_steps} | " + " | ".join(agent_states)
        )

        if dones.get("__all__", False):
            self.timer.stop()
            self.status_label.setText("Simulation finished.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HMIDemoWindow()
    window.show()
    sys.exit(app.exec())