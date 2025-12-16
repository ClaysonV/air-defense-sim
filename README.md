# Global Air Defense Simulator

**A Python-based kinematic simulation environment for modeling aerial intercepts and sensor fusion algorithms.**

## Project Overview

The Global Air Defense Simulator (GADS) is a physics-based engine designed to model the kinematic interactions between aerial targets and surface-to-air missile (SAM) systems. Unlike simplified gaming approximations, GADS utilizes mathematical models to simulate atmospheric flight dynamics, radar sensor noise relative to Radar Cross Section (RCS), and guidance control loops.

The primary objective of this project is to visualize the efficacy of Proportional Navigation (PN) guidance laws and Linear Kalman Filtering (LKF) in engagement scenarios involving varying degrees of target observability otherwise known as stealth.

## Simulation Interface
The simulation outputs a multi-panel data visualization dashboard for real-time analysis:
### 1. **3D Kinematic Display**
Visualizes the spatial relationship between the interceptor and the target in a 3D Cartesian coordinate system.
* **State Estimation:** Displays both the measured state (raw sensor returns with Gaussian noise) and the estimated state (Kalman Filter output), effectively visualizing the noise reduction algorithm.
* **Environmental Modeling:** Features a 3D terrain mesh and exponentially decaying atmospheric density visualization.
### 2. Sensor Modeling (PPI Scope)
Simulates a Plan Position Indicator (PPI) radar display.
* **Signal-to-Noise Representation:** Demonstrates the impact of RCS on sensor resolution. Targets with low RCS (e.g., F-22) generate high-variance measurement scatter, while high-RCS targets (e.g., B-52) produce coherent return signals.
### 3. Real-Time Telemetry
Provides a numeric readout of critical flight parameters, including:
* **Mach Number:** Dynamically calculated based on local speed of sound at altitude.
* **G-Load:** Lateral acceleration loads applied to the airframe.
* **System Status:** Guidance loop status (Searching vs. Locked).
### 4. Intercept Analytics ($P_k$)
A real-time probability graph plotting the Probability of Kill ($P_k$) as a function of range, closing velocity, and tracker stability.

## Tech Stack
* **Python 3.10+**
* **NumPy:** For high-performance vector calculus, matrix operations (Kalman Filtering), and physics calculations.
* **Matplotlib:** For rendering the real-time simulation dashboard, financial-style analytics graphs, and radar scopes
* **Mplot3d:** For creating the interactive 3D tactical environment and terrain wireframes.


## Installation

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/ClaysonV/air-defense-sim.git
    cd air-defense-sim
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Run the Engine**
    ```bash
    python interceptor.py
    ```

2. **Scenario Configuration**
The system will initialize the Asset Library. You will be prompted to configure the engagement parameters:
    * **Target Selection:** Choose from a database of fixed-wing aircraft, UAVs, and experimental concepts (parameterized by Cruise Speed, Max Speed, G-Limits, and RCS).
    * **Interceptor Selection:** Choose from a database of defensive systems (parameterized by Thrust, Burn Time, and Navigation Gain).

3. **Simulation Loop**

    The simulation runs in real-time steps ($dt=0.05s$). The terminal will output event logs (Launch Detection, Intercept, Splash) while the dashboard visualizes the guidance solution.

## Technical Specifications
### 1. **Atmospheric Model**
Air density ($\rho$) is modeled as an exponential decay function of altitude ($h$), influencing both drag forces and aerodynamic control authority.
* $$\rho(h) = \rho_0 e^{-h/H}$$
* $\rho_0$: Sea level air density ($1.225 \text{ kg/m}^3$).
* $H$: Scale height ($\approx 8,500 \text{ m}$).

### 2. **Aerodynamic Drag**

Energy management is governed by the drag equation, creating a realistic trade-off between kinetic energy and maneuverability.
* $$F_d = \frac{1}{2} \rho v^2 C_d A$$

### 3. **Guidance Law: Proportional Navigation (PN)**

The interceptor utilizes True Proportional Navigation (TPN). The guidance command generates lateral acceleration ($\vec{a}_c$) proportional to the Line-of-Sight (LOS) rotation rate.
* $$\vec{a}_c = N V_c \dot{\vec{\lambda}}$$
* $N$: Navigation Constant (Gain).
* $V_c$: Closing Velocity.
* $\dot{\lambda}$: LOS Rotation Rate.

### 4. **Signal Processing: Linear Kalman Filter**

To simulate electronic warfare environments, the system applies a Linear Kalman Filter (LKF) to estimate the state vector $\mathbf{x}$ (position and velocity) from noisy measurements $\mathbf{z}$.
The measurement noise covariance $R$ is dynamically scaled inversely to the target's Radar Cross Section (RCS):
* Low RCS: High Measurement Noise $\rightarrow$ Low Kalman Gain (Filter trusts model prediction).
* High RCS: Low Measurement Noise $\rightarrow$ High Kalman Gain (Filter trusts sensor data).

## Extensibility
The asset database is designed for modularity. Users may define custom airframes by modifying the JETS dictionary in interceptor.py:
```bash
# Example: Adding a custom test platform
"99": {
    "name": "X-Test Platform",
    "cr_spd": 500,    # Cruise Speed (m/s)
    "max_spd": 900,   # Max Speed (m/s)
    "g": 15.0,        # G-Limit
    "rcs": 0.01,      # Radar Cross Section (m^2)
    "col": "purple"   # Visualization Color
},
```
