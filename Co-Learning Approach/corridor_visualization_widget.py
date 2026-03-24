"""
Corridor Visualization Widget
Features:
- Grid visualization with tracks
- Agents as colored rectangles with labels
- Current positions and goals shown
- Auto-scaling and clean rendering
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


class CorridorVisualizationWidget(QWidget):
    """Widget to visualize the corridor environment"""
    
    def __init__(self, env=None, parent=None):
        super().__init__(parent)
        self.env = env
        self.show_conflicts = False  # Toggle for showing conflict zones
        self.conflict_positions = []  # List of (row, col) tuples for conflicts
        
        # Setup matplotlib figure
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Agent colors
        self.agent_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#F38181']
        
        # Initial render
        if self.env:
            self.update_visualization()
    
    def set_environment(self, env):
        """Set the environment to visualize"""
        self.env = env
        self.update_visualization()
    
    def set_conflict_positions(self, positions):
        """Set positions where conflicts are detected"""
        self.conflict_positions = positions if positions else []
        print(f"DEBUG VIZ: set_conflict_positions called with: {self.conflict_positions}")
        print(f"DEBUG VIZ: show_conflicts is: {self.show_conflicts}")
        if self.show_conflicts:
            print("DEBUG VIZ: Updating visualization to show conflicts...")
            self.update_visualization()
    
    def toggle_conflicts(self, show):
        """Toggle conflict highlighting on/off"""
        self.show_conflicts = show
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization with current environment state"""
        if not self.env:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get environment state
        rows = self.env.rows
        cols = self.env.cols
        grid = self.env.grid
        
        # Draw grid
        self._draw_grid(ax, rows, cols, grid)
        
        # Draw agents
        self._draw_agents(ax)
        
        # Draw goals
        self._draw_goals(ax)
        
        # Draw conflict zones if enabled
        if self.show_conflicts:
            self._draw_conflicts(ax)
        
        # Configure plot - zoom out more for better view
        margin = 1.5  # Add margin around grid
        ax.set_xlim(-margin, cols - 1 + margin)
        ax.set_ylim(rows - 1 + margin, -margin)  # Inverted Y for top-down view
        ax.set_aspect('equal')
        ax.set_xticks(range(0, cols, 5))
        ax.set_yticks(range(rows))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Column', fontsize=10)
        ax.set_ylabel('Row', fontsize=10)
        
        # Simple clean title
        title = 'Compact 3×5 Junction - Multi-Agent Railway Coordination'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Refresh canvas
        self.canvas.draw()
    
    def _draw_grid(self, ax, rows, cols, grid):
        """Draw the railway grid"""
        # Track types: 0=empty, 1=horizontal, 2=vertical, 3=junction
        
        for r in range(rows):
            for c in range(cols):
                track_type = grid[r, c]
                
                if track_type == 0:
                    # Empty - light gray background
                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor='#F5F5F5'
                    )
                    ax.add_patch(rect)
                
                elif track_type == 1:
                    # Horizontal track
                    ax.plot([c - 0.4, c + 0.4], [r, r], 'k-', linewidth=3)
                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor='#E8E8E8'
                    )
                    ax.add_patch(rect)
                
                elif track_type == 2:
                    # Vertical track
                    ax.plot([c, c], [r - 0.4, r + 0.4], 'k-', linewidth=3)
                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor='#E8E8E8'
                    )
                    ax.add_patch(rect)
                
                elif track_type == 3:
                    # Junction - both horizontal and vertical
                    ax.plot([c - 0.4, c + 0.4], [r, r], 'k-', linewidth=3)
                    ax.plot([c, c], [r - 0.4, r + 0.4], 'k-', linewidth=3)
                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0, facecolor='#D8D8D8'
                    )
                    ax.add_patch(rect)
    
    def _draw_agents(self, ax):
        """Draw agents as colored rectangles with labels and direction arrows"""
        if not hasattr(self.env, 'agents_pos'):
            return
        
        for i in range(self.env.n_agents):
            if i >= len(self.env.agents_pos):
                break
            
            pos = self.env.agents_pos[i]
            r, c = pos[0], pos[1]
            
            # Get agent color
            color = self.agent_colors[i % len(self.agent_colors)]
            
            # Check if agent is done (reached goal)
            is_done = self.env.agents_done[i] if i < len(self.env.agents_done) else False
            alpha = 0.5 if is_done else 0.9  # Fade out completed agents
            edge_color = 'green' if is_done else 'black'  # Green border if done
            edge_width = 3 if is_done else 2
            
            # Draw agent as SMALLER rectangle (0.25 instead of 0.35)
            train_size = 0.25
            agent_rect = patches.Rectangle(
                (c - train_size, r - train_size), train_size * 2, train_size * 2,
                linewidth=edge_width, edgecolor=edge_color, facecolor=color, alpha=alpha
            )
            ax.add_patch(agent_rect)
            
            # Add agent label (smaller font)
            label_color = 'white' if not is_done else 'darkgreen'
            ax.text(c, r, f'{i}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=label_color)
            
            # Add checkmark if done
            if is_done:
                ax.text(c + 0.4, r - 0.4, '✓', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='green')
    
    def _draw_goals(self, ax):
        """Draw agent goals as colored star markers (no labels)"""
        if not hasattr(self.env, 'agent_configs'):
            return
        
        for i in range(self.env.n_agents):
            if i >= len(self.env.agent_configs):
                break
            
            config = self.env.agent_configs[i]
            goal_r, goal_c = config[2], config[3]
            
            # Get agent color
            color = self.agent_colors[i % len(self.agent_colors)]
            
            # Draw goal as star - clean, no labels
            ax.plot(goal_c, goal_r, marker='*', markersize=25,
                   color=color, markeredgecolor='black', markeredgewidth=2,
                   zorder=10)
    
    def _draw_conflicts(self, ax):
        """Draw red circles around conflict positions"""
        print(f"DEBUG VIZ: _draw_conflicts called! conflict_positions = {self.conflict_positions}")
        
        if not self.conflict_positions:
            print("DEBUG VIZ: No conflict positions to draw!")
            return
        
        for pos in self.conflict_positions:
            r, c = pos
            print(f"DEBUG VIZ: Drawing circle at position ({r}, {c})")
            
            # Draw red circle around conflict zone
            circle = patches.Circle(
                (c, r), 0.6,  # Radius 0.6 to be visible but not huge
                linewidth=3,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                alpha=0.8,
                zorder=25  # Draw on top
            )
            ax.add_patch(circle)
            
            # Add warning symbol
            ax.text(c, r - 0.8, '⚠️', ha='center', va='center',
                   fontsize=16, zorder=26)
        
        print(f"DEBUG VIZ: Finished drawing {len(self.conflict_positions)} conflict circles")


if __name__ == "__main__":
    """Test visualization with dummy environment"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    # Create dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.rows = 12
            self.cols = 35
            self.n_agents = 3
            self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
            
            # Main corridor (Row 5)
            for c in range(self.cols):
                self.grid[5, c] = 1
            
            # Vertical branches
            for r in range(1, 10):
                self.grid[r, 7] = 2
                self.grid[r, 12] = 2
                self.grid[r, 27] = 2
            
            # Bypasses
            for c in range(1, self.cols):
                if self.grid[3, c] != 2:
                    self.grid[3, c] = 1
                if self.grid[7, c] != 2:
                    self.grid[7, c] = 1
            
            # Junctions
            for r, c in [(5, 7), (5, 12), (5, 27), (3, 7), (3, 12), (3, 27), (7, 7), (7, 12), (7, 27)]:
                self.grid[r, c] = 3
            
            # Agents
            self.agents_pos = [[5, 1], [5, 33], [1, 12]]
            self.agents_dir = [1, 3, 2]
            self.agents_done = [False, False, False]
            self.agent_configs = [
                (5, 1, 5, 33, 1, 0),
                (5, 33, 5, 1, 3, 0),
                (1, 12, 9, 27, 2, 5),
            ]
    
    app = QApplication(sys.argv)
    env = DummyEnv()
    widget = CorridorVisualizationWidget(env)
    widget.setWindowTitle("Corridor Visualization Test")
    widget.resize(1000, 600)
    widget.show()
    sys.exit(app.exec())