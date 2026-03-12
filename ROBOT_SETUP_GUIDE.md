# UR5 Robot Setup and Configuration Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installing ikpy IK Solver](#installing-ikpy-ik-solver)
- [Basic Configuration](#basic-configuration)
- [Testing IK Solver](#testing-ik-solver)
- [Safety Limits and Configuration](#safety-limits-and-configuration)
- [Gripper Control](#gripper-control)
- [Trajectory Smoothing](#trajectory-smoothing)
- [Real Robot Testing Checklist](#real-robot-testing-checklist)

---

## Prerequisites

### Hardware Requirements

1. **UR5 Collaborative Robot**
   - Universal Robots UR5 (or compatible 6-DOF arm)
   - Network-connected to your computer
   - IP address: typically `192.168.1.100` or check network panel

2. **Network Configuration**
   - Robot and PC on same network
   - Ping test: `ping 192.168.1.100` should respond
   - No firewall blocking port 30003 (RTDE) or port 30002 (secondary client interface)

3. **Physical Setup**
   - Robot mounted securely on table or pedestal
   - Workspace clear of obstacles
   - Emergency stop button accessible
   - Manual teach pendant nearby

### Software Requirements

- Python 3.7+
- Dependencies: numpy, scipy, ikpy
- Linux, macOS, or Windows

---

## Installing ikpy IK Solver

### Step 1: Install ikpy

```bash
# Install from pip
pip install ikpy scipy numpy

# Verify installation
python -c "import ikpy; print(ikpy.__version__)"
```

### Step 2: Verify UR5 DH Parameters

ikpy needs Denavit-Hartenberg (DH) parameters for UR5. These are built into `robot_controller.py`:

```python
from robot_controller import RobotArmUR5

ur5 = RobotArmUR5()

# DH parameters are pre-configured:
print(ur5.dh_params)
# Output:
# {
#   'd': [0.08916, 0, 0, 0.10915, 0.09475, 0.0823],
#   'a': [0, -0.425, -0.39225, 0, 0, 0],
#   'alpha': [π/2, 0, 0, π/2, -π/2, 0],
# }
```

**Note:** These values are standard UR5 dimensions. If using a modified arm, contact Universal Robots for exact parameters.

---

## Basic Configuration

### Step 1: Configure Robot IP Address

Edit `config.py`:

```python
# ===== ROBOT CONFIGURATION =====
robot_ip = "192.168.1.100"          # UR5 IP address
robot_port = 30003                   # RTDE port (Real-Time Data Exchange)
robot_connection_timeout = 5.0       # Seconds to wait for connection
enable_robot_control = False         # Set to True to enable
```

### Step 2: Find Your Robot's IP Address

**On UR5 Control Pendant:**
```
Main Menu → System → About → IP Address
```

Or from network:
```bash
# On Linux/macOS
nmap -p 30003 192.168.1.0/24  # Scan subnet for UR5

# On Windows (if nmap not available)
arp -a  # List all devices on network
```

### Step 3: Enable Python Script Runtime

On UR5 pendant:
```
Main Menu → System → Settings → Safety → Allow Remote Control: ON
Main Menu → System → Settings → Communication → RTDE Port: 30003
```

**Important:** For gesture control to work, you must enable remote control.

---

## Testing IK Solver

### Step 1: Test Without Robot

```bash
python test_robot_ik.py
```

This tests IK solver offline - no actual robot connection needed.

**Example output:**
```
========================================
ROBOT IK SOLVER TEST
========================================

[Test 1] Forward reach (0.5, 0, 0.5)
  IK Solution: [0.00, -1.57, 1.57, 0.00, 0.00, 0.00]
  Status: [OK]

[Test 2] Workspace boundary
  IK Solution: [0.79, -1.57, 1.57, 0.00, 0.00, 0.00]
  Status: [OK]

[Test 3] Unreachable position
  IK Solution: None (out of workspace)
  Status: [OK] - correctly rejected

Result: 3/3 tests passed - IK solver is functional
```

### Step 2: Run Example Script

```bash
python example_robot_arm_control.py
```

This runs a full gesture→robot mapping simulation. Watch output for:
- IK solution success rate (target: >80% for reachable targets)
- Reachability analysis showing which areas are workspace
- Gripper command counts (pinch→close, open→open)
- Performance: latency <15ms per IK solve

**Expected output excerpt:**
```
Frame   0: Gesture=reach_forward    Reach=100.0% IK_OK=  1 IK_Fail=  0 Latency= 5.23ms
Frame  60: Gesture=reach_forward    Reach= 99.8% IK_OK= 61 IK_Fail=  0 Latency= 4.87ms
Frame 120: Gesture=circular_motion  Reach= 93.5% IK_OK=120 IK_Fail=  1 Latency= 5.15ms
...

[IK Solver Performance]
  Successful IK solutions: 298
  IK failures/unreachable: 2
  Success rate: 99.3%
```

---

## Safety Limits and Configuration

### Joint Angle Limits

**Standard UR5 limits (from vendor):**
```python
joint_limits = {
    "lower": [-π, -π, -π, -π, -π, -π],    # All joints: -180°
    "upper": [π,  π,  π,  π,  π,  π],     # All joints: +180°
}
```

These are enforced in `RobotArmUR5.inverse_kinematics()`.

### Velocity Limits

```python
# In HandToArmMapper:
max_velocity = 0.5  # meters/second
```

Gesture hand speed is scaled to arm speed. If hand moves at 1.5 m/s, arm will be commanded at 0.5 m/s maximum.

**Adjust if needed:**
```python
from robot_controller import HandToArmMapper

mapper = HandToArmMapper(
    position_scale=1.0,
    max_velocity=0.3  # Slower for safety
)
```

### Acceleration Limits

UR5 has built-in acceleration limits (from robot controller). These are NOT configurable from Python - set on pendant:
```
Main Menu → Program → Move → Acceleration: [set value]
```

### Collision Detection (Optional)

For advanced safety, enable PyBullet collision checking:

```python
# In config.py
robot_enable_collision_check = True

# Then in robot_controller.py, collision checking is enabled during
# trajectory planning. Requires: pip install pybullet
```

**Note:** Collision checking adds 50-100ms per query. Use only if necessary.

---

## Gripper Control

### Configuration

UR5 can use different grippers (Robotiq 2F-85, OnRobot VGC, etc.). Configuration depends on gripper model:

```python
from robot_controller import GripperCommand

# Basic gripper commands
gripper_open = GripperCommand.OPEN      # Full open (0% force)
gripper_close = GripperCommand.CLOSE    # Full close (100% force)
gripper_stop = GripperCommand.STOP      # Hold current position
```

### Gesture Mapping

Default mapping (in `object_manipulator.py`):
```python
if gesture == "pinch":
    gripper_command = GripperCommand.CLOSE
elif gesture == "open":
    gripper_command = GripperCommand.OPEN
```

### Setting Gripper Force

Gripper closing force (0-100 scale):

```python
from robot_controller import RobotTarget, GripperCommand

target = RobotTarget(
    position=np.array([0.5, 0, 0.5]),
    orientation=np.array([0, 0, 0]),
    gripper_command=GripperCommand.CLOSE,
    gripper_force=75.0  # 75% force (0-100 scale)
)

robot.move_to_position(target)
```

### Testing Gripper

```bash
# Manual gripper test
python -c "
from robot_controller import RobotArmUR5, RobotTarget, GripperCommand
import numpy as np

robot = RobotArmUR5('192.168.1.100')
robot.connect()

# Test open
robot.move_to_position(RobotTarget(
    position=np.array([0.5, 0, 0.5]),
    gripper_command=GripperCommand.OPEN
))
print('Gripper opened')

import time
time.sleep(2)

# Test close
robot.move_to_position(RobotTarget(
    position=np.array([0.5, 0, 0.5]),
    gripper_command=GripperCommand.CLOSE,
    gripper_force=50.0
))
print('Gripper closed at 50% force')
"
```

---

## Trajectory Smoothing

### Issue: Jerky Robot Motion

**Symptom:** Robot arm moves in discrete jumps instead of smooth curves

**Cause:** Hand coordinates have jitter or update irregularly

**Solution:** Use `HandToArmMapper.smooth_trajectory()`:

```python
from robot_controller import HandToArmMapper

mapper = HandToArmMapper(
    position_scale=1.0,
    max_velocity=0.5          # Limits speed
)

# In main loop:
current_target = current_state.robot_position
gesture_target = extract_hand_position()

smoothed_target = mapper.smooth_trajectory(
    current=current_target,
    target=gesture_target,
    dt=0.033,                 # 30 fps frame time
    max_velocity=0.5          # Also rate-limits
)

robot.move_to_position(smoothed_target)
```

### Tuning Smoothing

**If motion is too jerky:** Reduce dt (increase update rate) or reduce max_velocity:
```python
smoothed_target = mapper.smooth_trajectory(
    current, target,
    dt=0.016,  # 60 fps instead of 30 fps
    max_velocity=0.3  # Slower motion
)
```

**If motion is too slow:** Increase max_velocity:
```python
smoothed_target = mapper.smooth_trajectory(
    current, target,
    dt=0.033,
    max_velocity=0.8  # Faster (still safe - under 1.0 m/s)
)
```

---

## Real Robot Testing Checklist

### Before First Motion

- [ ] Robot arm is in **Safe Mode** on pendant
- [ ] Area around robot is clear (1.5m radius)
- [ ] Emergency stop button is accessible and tested
- [ ] Manual teach pendant is in operator's hand
- [ ] No people in robot workspace
- [ ] Gripper has no objects in it
- [ ] IK solver tested offline (`python test_robot_ik.py`)
- [ ] Robot connection test passed (`ping 192.168.1.100`)

### First Motion Test (Manual Speed)

1. **Set safe speed limit on pendant:**
   ```
   Main Menu → Program → Move → Speed: 10% of max
   ```

2. **Run slow gesture command:**
   ```python
   from robot_controller import RobotArmUR5, RobotTarget
   import numpy as np

   robot = RobotArmUR5("192.168.1.100")
   if not robot.connect():
       print("Connection failed!")
       exit()

   # Safe target: close to home position
   target = RobotTarget(position=np.array([0.4, 0, 0.6]))

   print("Sending first command (SLOW - 10% on pendant)...")
   success = robot.move_to_position(target)
   print(f"Result: {success}")
   ```

3. **Observe:**
   - Arm moves slowly and smoothly
   - Reaches target without jerking
   - Stops cleanly
   - No warning lights/buzzing

4. **Increase speed gradually:**
   - 10% → 25% → 50% → 100% (over several tests)
   - Watch for unexpected behavior at each step

### Second Phase: Gesture Integration

1. **Test single gesture** (e.g., hand reach):
   ```bash
   python example_robot_arm_control.py
   ```
   (Watch first 10 frames, then stop with Ctrl+C)

2. **Enable gesture mode slowly:**
   - Start with slow hand motion
   - Increase speed gradually
   - Watch that IK solutions are valid

3. **Test emergency stop:**
   - Press E-stop button
   - Arm should stop immediately
   - Reset pendant to resume

### Performance Benchmarking

Once comfortable, measure actual performance:

```bash
python test_robot_ik.py   # Should see <15ms per IK solve

python example_robot_arm_control.py | grep "Latency\|IK_Fail"
# Target: Most frames <30ms, IK success >95%
```

---

## Troubleshooting

### "Connection refused on port 30003"

**Cause:** Robot firewall, wrong IP, or RTDE disabled

**Fix:**
```bash
# Check connectivity
ping 192.168.1.100

# On UR5 pendant, check:
# Settings → Communication → RTDE Port Enable: ON
# Settings → Safety → Allow Remote Control: ON
```

### "IK solver finds no solution"

**Cause:** Position is outside UR5's 850mm reach

**Fix:**
- Use UR5 workspace model to validate positions
- Check that coordinates are in meters (not mm)
- Visualize with `test_robot_ik.py` reachability plot

### "Gripper not opening/closing"

**Cause:** Wrong gripper model, wrong command interface

**Fix:**
- Check gripper type on hardware
- Verify gripper control channel (could be I/O, not RTDE)
- Test with manual pendant control first

### "Arm jerks or moves unexpectedly"

**Cause:** IK solution switches between elbow-up and elbow-down configurations

**Fix:**
- Use `smooth_trajectory()` as described above
- Reduce hand motion speed
- Increase trajectory smoothing factor

---

## Performance Tuning

| Parameter | Default | Fast | Safe |
|-----------|---------|------|------|
| max_velocity (m/s) | 0.5 | 0.8 | 0.2 |
| IK solver max_iter | 1000 | 100 | 2000 |
| smooth_trajectory dt | 0.033 | 0.016 | 0.05 |
| gripper_force (%) | 50 | 80 | 30 |

**Recommendation:** Start with "Safe" column, gradually move to "Fast" as you gain confidence.

---

## Summary

1. **Setup:** Install ikpy, configure IP, test IK solver offline
2. **Safety:** Understand joint limits, velocity limits, collision detection
3. **Testing:** Follow checklist before any real motion
4. **Tuning:** Smooth trajectory, adjust speeds gradually
5. **Maintenance:** Regularly test emergency stop, keep workspace clear

Once operational, the system will maintain 20-30 FPS gesture→robot mapping with <50ms latency including network.
