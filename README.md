# Quadruped Robot for Autonomous Locomotion and Obstacle Navigation

This project focuses on the development of a quadruped robot designed for autonomous locomotion in an aisle-like environment. The robot is equipped with wheels on its feet, allowing it to both walk and drive, with the ability to switch modes depending on the terrain and obstacles it encounters.

## Features
- **Hybrid Locomotion:** The robot can walk using its four legs or drive using its foot-mounted wheels, switching between modes to optimize mobility.
- **Real-Time Obstacle Avoidance:** Equipped just with a camera, the robot can detect and navigate around obstacles in real time.
- **Autonomous Pathfinding:** Using algorithms for path planning, the robot can find the most efficient route to its destination while avoiding obstacles.
- **Multi-Joint Coordination:** Each leg has two joints, providing a range of movement to adapt to different terrains and challenges.
- **Terrain-Based Mode Switching:** The robot autonomously decides when to walk (e.g., over complex obstacles) or drive (e.g., smooth surfaces).

## Core Components
- **Sensor Processing:** Processes data from LIDAR and cameras to understand the environment and make navigation decisions.
- **Locomotion Control:** Coordinates the legs and wheels for smooth transitions between walking and driving.
- **Pathfinding Algorithm:** Implements algorithms for obstacle avoidance and efficient path planning.

## Installation
To install the required dependencies:
```bash
pip install -r requirements.txt
