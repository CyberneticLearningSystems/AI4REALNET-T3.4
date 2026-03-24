import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.agent_utils import TrainState

from src.configs.ControllerConfigs import PPOControllerConfig
from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size, obs_dict_to_tensor
from src.utils.observation.normalisation import FlatlandNormalisation
from src.environments.scenario_loader import load_scenario_from_json
from src.hmi.human_input import HumanInputWidget


class FlatlandHMIApp(QWidget):
    """
    HMI + Flatland + PPO demo app for T3.4.

    Design:
    - HMI does not directly create final actions.
    - PPO controller remains the main decision layer.
    - HMI currently stores a structured command/context that can later
      be injected into the policy input or used for policy conditioning.
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AI4REALNET T3.4 HMI + PPO")
        self.resize(1280, 850)

        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------
        self.device = torch.device("cpu")
        self.use_fallback_policy = False

        # Scenario to visualise
        self.scenario_path = Path("src/environments/simple_avoidance.json")

        # PPO config used to construct the controller
        # This config contains encoder / actor / critic structure.
        self.config_path = "src/configs/PPO_FNN.yaml"

        self.model_dir = "models/FNN_large_env_run1"

        self.max_steps = 200
        self.timer_interval_ms = 500

        # ------------------------------------------------------------
        # Load PPO controller config
        # ------------------------------------------------------------
        self.config = load_config_file(self.config_path)
        self.controller = self._build_controller(self.config)
        self.normaliser = None

        # Observation builder settings are taken from the PPO config
        obs_cfg = self.config["environment_config"]["observation_builder_config"]
        if obs_cfg["type"] != "tree":
            raise ValueError("This HMI app currently expects a tree observation PPO setup.")

        self.max_depth = obs_cfg["max_depth"]

        # ------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------
        self.env = load_scenario_from_json(
            str(self.scenario_path),
            observation_builder=TreeObsForRailEnv(max_depth=self.max_depth),
        )
        self.obs, self.info = self.env.reset()

        self.normaliser = FlatlandNormalisation(
            n_nodes=self.controller.config["n_nodes"],
            n_features=self.controller.config["n_features"],
            n_agents=self.env.number_of_agents,
            env_size=(self.env.width, self.env.height),
        )

        self._try_load_model(self.model_dir)

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.step_count = 0
        self.last_rewards: Dict[int, float] = {}
        self.active_hmi_command: Optional[Dict[str, Any]] = None

        # Gate-style HMI behaviour:
        # an agent is released once the reference agent has progressed far enough
        self.release_distance = 4

        # Hard STOP remains separate
        self.delay_targets: Dict[int, int] = {}
        self.priority_targets: Dict[int, int] = {}
        self.agent_progress: Dict[int, int] = {}

        # ------------------------------------------------------------
        # Renderer
        # ------------------------------------------------------------
        self.renderer = RenderTool(
            self.env,
            gl="PILSVG",
            screen_width=1200,
            screen_height=600,
        )

        # ------------------------------------------------------------
        # UI
        # ------------------------------------------------------------
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.title_label = QLabel("T3.4 HMI with PPO Controller")
        self.scenario_label = QLabel(f"Scenario: {self.scenario_path.name}")
        self.status_label = QLabel("Ready.")
        self.hmi_label = QLabel("HMI command: None")
        self.reward_label = QLabel("Rewards: {}")
        self.model_label = QLabel(f"Model dir: {self.model_dir}")

        self.map_label = QLabel(self)
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        agent_handles = list(range(self.env.get_num_agents()))
        self.human_input = HumanInputWidget(agent_handles=agent_handles)
        self.human_input.tokens_signal.connect(self.on_tokens_selected)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        self.clear_hmi_button = QPushButton("Clear HMI")

        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.clear_hmi_button.clicked.connect(self.clear_hmi_command)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.clear_hmi_button)

        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.human_input)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addWidget(self.scenario_label)
        self.main_layout.addWidget(self.model_label)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.hmi_label)
        self.main_layout.addWidget(self.reward_label)
        self.main_layout.addWidget(self.map_label)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_step)

        self.render_env_to_label()
        self.print_scenario_info()

    # ------------------------------------------------------------------
    # Controller / PPO setup
    # ------------------------------------------------------------------
    def _build_controller(self, config: Dict[str, Any]):
        controller_cfg = copy.deepcopy(config["controller_config"])
        n_nodes, state_size = calculate_state_size(
            config["environment_config"]["observation_builder_config"]["max_depth"]
        )
        controller_cfg["n_nodes"] = n_nodes
        controller_cfg["state_size"] = state_size

        controller = PPOControllerConfig(controller_cfg).create_controller()
        controller.to(self.device)
        controller.eval()
        return controller

    def _try_load_model(self, model_dir: str) -> None:
        actor_path = os.path.join(model_dir, "actor.pth")
        critic_path = os.path.join(model_dir, "critic.pth")
        encoder_path = os.path.join(model_dir, "encoder.pth")

        if os.path.exists(actor_path) and os.path.exists(critic_path) and os.path.exists(encoder_path):
            print(f"[model] loading PPO checkpoint from {model_dir}")
            self.controller.actor_network.load_state_dict(
                torch.load(actor_path, map_location=self.device, weights_only=True)
            )
            self.controller.critic_network.load_state_dict(
                torch.load(critic_path, map_location=self.device, weights_only=True)
            )
            self.controller.encoder_network.load_state_dict(
                torch.load(encoder_path, map_location=self.device, weights_only=True)
            )
            print("[model] checkpoint loaded")
        else:
            print(f"[model] no complete checkpoint found in {model_dir}")
            print("[model] running with fallback policy (constant forward)")
            self.use_fallback_policy = True

    # ------------------------------------------------------------------
    # UI controls
    # ------------------------------------------------------------------
    def start_simulation(self) -> None:
        if not self.timer.isActive():
            self.timer.start(self.timer_interval_ms)
            self.status_label.setText("Simulation running...")

    def pause_simulation(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Simulation paused.")

    def reset_simulation(self) -> None:
        if self.timer.isActive():
            self.timer.stop()

        self.obs, self.info = self.env.reset()
        self.step_count = 0
        self.last_rewards = {}
        self.active_hmi_command = None
        self.delay_targets = {}
        self.priority_targets = {}
        self.agent_progress = {}

        self.render_env_to_label()
        self.status_label.setText("Simulation reset.")
        self.hmi_label.setText("HMI command: None")
        self.reward_label.setText("Rewards: {}")

        print("[reset] environment reset")

    def clear_hmi_command(self) -> None:
        self.active_hmi_command = None
        self.delay_targets = {}
        self.priority_targets = {}
        self.agent_progress = {}
        self.hmi_label.setText("HMI command: None")
        self.status_label.setText("Cleared HMI command.")
        print("[HMI] cleared active command")

    # ------------------------------------------------------------------
    # HMI handling
    # ------------------------------------------------------------------
    def on_tokens_selected(self, tokens: Dict[int, Any]) -> None:
        print("\n=== TOKEN RECEIVED ===")
        print(tokens)

        self.active_hmi_command = self.tokens_to_command(tokens)

        print("[HMI COMMAND]", self.active_hmi_command)

        if self.active_hmi_command:
            action = self.active_hmi_command.get("action")
            primary = self.active_hmi_command.get("primary_agent")
            secondary = self.active_hmi_command.get("secondary_agent")

            if action == "DELAY" and primary is not None:
                # Delay agent waits until the "other" agent has progressed enough
                other_agents = [a for a in range(self.env.number_of_agents) if a != primary]
                if other_agents:
                    self.delay_targets[primary] = other_agents[0]
                    print(
                        f"[HMI] DELAY applied: agent {primary} waits until "
                        f"agent {other_agents[0]} has progressed by at least {self.release_distance} cells"
                    )

            elif action == "STOP" and primary is not None:
                print(f"[HMI] STOP applied: agent {primary} stays stopped until reset/clear")

            elif action == "PRIORITY" and primary is not None and secondary is not None:
                self.priority_targets[secondary] = primary
                print(
                    f"[HMI] PRIORITY applied: secondary agent {secondary} waits until "
                    f"primary agent {primary} has progressed by at least {self.release_distance} cells"
                )

        self.hmi_label.setText(f"HMI command: {self.active_hmi_command}")
        self.status_label.setText("HMI command updated.")

    def tokens_to_command(self, tokens: Dict[int, Any]) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None

        action = tokens.get(0)
        primary = tokens.get(1)
        secondary = tokens.get(2)

        def safe_int(value: Any) -> Optional[int]:
            if value in (None, "", "None"):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        command: Dict[str, Any] = {
            "action": None,
            "primary_agent": safe_int(primary),
            "secondary_agent": safe_int(secondary),
        }

        if action == "Prioritise":
            command["action"] = "PRIORITY"
        elif action == "Stop":
            command["action"] = "STOP"
        elif action == "Delay":
            command["action"] = "DELAY"
        else:
            command["action"] = action

        return command

    def build_hmi_context(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "command": self.active_hmi_command,
        }
    
    def get_agent_position(self, agent_handle: int):
        """
        Return the current position of the agent if available,
        otherwise fall back to initial position / None.
        """
        agent = self.env.agents[agent_handle]

        if getattr(agent, "position", None) is not None:
            return agent.position

        if getattr(agent, "initial_position", None) is not None:
            return agent.initial_position

        return None

    def manhattan_distance(self, pos_a, pos_b) -> int:
        if pos_a is None or pos_b is None:
            return 0
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def has_progressed_enough(self, moving_agent: int) -> bool:
        """
        Release condition:
        Agent is considered 'done' once it reached its target.
        """
        agent = self.env.agents[moving_agent]
        return agent.state == TrainState.DONE

    def update_agent_progress(self) -> None:
        """
        Store the maximum progress reached so far for each agent.
        This avoids losing progress information when an agent position becomes None.
        """
        for agent_handle in range(self.env.number_of_agents):
            agent = self.env.agents[agent_handle]
            start_pos = getattr(agent, "initial_position", None)
            current_pos = getattr(agent, "position", None)

            if start_pos is None:
                continue

            if current_pos is None:
                continue

            progress = self.manhattan_distance(start_pos, current_pos)

            old_progress = self.agent_progress.get(agent_handle, 0)
            self.agent_progress[agent_handle] = max(old_progress, progress)

    # ------------------------------------------------------------------
    # PPO path
    # ------------------------------------------------------------------
    def prepare_policy_input(
        self,
        obs: Any,
        hmi_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Convert Flatland observation dict to PPO-ready tensor.
        """
        obs_tensor = obs_dict_to_tensor(
            observation=obs,
            obs_type="tree",
            n_agents=self.env.number_of_agents,
            max_depth=self.max_depth,
            n_nodes=self.controller.config["n_nodes"],
        ).float()

        obs_tensor = self.normaliser.normalise(obs_tensor.unsqueeze(0)).squeeze(0)
        obs_tensor = obs_tensor.to(self.device)
        return obs_tensor

    def get_policy_actions(
        self,
        policy_obs: torch.Tensor,
        hmi_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, int]:
        """
        PPO remains the base action source.

        HMI commands are applied afterwards:
        - STOP: hard stop until reset / clear
        - DELAY: selected agent waits until the other agent has progressed enough
        - PRIORITY: secondary waits until primary has progressed enough
        """
        # ----------------------------------------------------------
        # Base policy (PPO or fallback)
        # ----------------------------------------------------------
        if getattr(self, "use_fallback_policy", False):
            action_dict = {
                agent: 2 for agent in range(self.env.number_of_agents)
            }
        else:
            with torch.no_grad():
                actions, _ = self.controller.select_action(policy_obs)

            action_values = np.atleast_1d(actions.detach().cpu().numpy())
            action_dict = {
                agent: int(action_values[agent])
                for agent in range(self.env.number_of_agents)
            }

        # ----------------------------------------------------------
        # Delay gates
        # ----------------------------------------------------------
        finished_delay_agents = []
        for waiting_agent, reference_agent in self.delay_targets.items():
            if self.has_progressed_enough(reference_agent):
                finished_delay_agents.append(waiting_agent)
            else:
                action_dict[waiting_agent] = 0

        for agent in finished_delay_agents:
            del self.delay_targets[agent]
            print(f"[HMI] DELAY released for agent {agent}")

        # ----------------------------------------------------------
        # Priority gates
        # ----------------------------------------------------------
        finished_priority_agents = []
        for secondary_agent, primary_agent in self.priority_targets.items():
            if self.has_progressed_enough(primary_agent):
                finished_priority_agents.append(secondary_agent)
            else:
                action_dict[secondary_agent] = 0

        for agent in finished_priority_agents:
            del self.priority_targets[agent]
            print(f"[HMI] PRIORITY gate released for secondary agent {agent}")

        # ----------------------------------------------------------
        # Hard STOP layer
        # ----------------------------------------------------------
        if hmi_context and hmi_context.get("command"):
            cmd = hmi_context["command"]
            action = cmd.get("action")
            primary = cmd.get("primary_agent")

            if action == "STOP" and primary is not None:
                action_dict[primary] = 0

        return action_dict

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_step(self) -> None:
        hmi_context = self.build_hmi_context()
        policy_obs = self.prepare_policy_input(self.obs, hmi_context)

        # Update progress before applying HMI gates
        self.update_agent_progress()

        actions = self.get_policy_actions(policy_obs, hmi_context)

        self.obs, rewards, done, self.info = self.env.step(actions)
        self.last_rewards = rewards
        self.step_count += 1

        self.render_env_to_label()

        self.status_label.setText(
            f"Step {self.step_count} | Actions: {actions}"
        )
        self.reward_label.setText(f"Rewards: {rewards}")

        print(
            f"[STEP {self.step_count}] "
            f"hmi={self.active_hmi_command} actions={actions} rewards={rewards}"
        )

        if done.get("__all__", False) or self.step_count >= self.max_steps:
            self.timer.stop()
            self.status_label.setText(f"Finished at step {self.step_count}.")
            print("[done] episode finished")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_env_to_label(self) -> None:
        self.renderer.render_env(show=False, show_observations=False)
        img = self.renderer.get_image()

        if not isinstance(img, np.ndarray):
            raise TypeError("Renderer returned a non-numpy image.")

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        img = np.ascontiguousarray(img)

        if img.ndim != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        h, w, ch = img.shape

        if ch == 4:
            fmt = QImage.Format.Format_RGBA8888
        elif ch == 3:
            fmt = QImage.Format.Format_RGB888
        else:
            raise ValueError(f"Unexpected number of channels: {ch}")

        bytes_per_line = ch * w
        qimg = QImage(img.tobytes(), w, h, bytes_per_line, fmt)
        pixmap = QPixmap.fromImage(qimg)

        self.map_label.setPixmap(
            pixmap.scaled(
                1100,
                520,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def print_scenario_info(self) -> None:
        print("\n=== SCENARIO INFO ===")
        print("Scenario:", self.scenario_path)
        print("Agents:", self.env.get_num_agents())

        for i, agent in enumerate(self.env.agents):
            print(f"Agent {i}:")
            print(f"  start:              {agent.initial_position}")
            print(f"  target:             {agent.target}")
            print(f"  earliest_departure: {getattr(agent, 'earliest_departure', None)}")

        print("======================\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlatlandHMIApp()
    window.show()
    sys.exit(app.exec())