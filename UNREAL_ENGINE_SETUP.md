# Unreal Engine 5 Setup Guide

## Quick Start

1. Enable Python plugin in UE5 editor
2. Create hand skeletal mesh (or import from marketplace)
3. Run `python example_unreal_integration.py`
4. Observe real-time hand skeleton animation in editor

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Enable Python Plugin](#enable-python-plugin)
- [Create Hand Skeleton Asset](#create-hand-skeleton-asset)
- [Setup Hand Animation Blueprint](#setup-hand-animation-blueprint)
- [Python API Connection](#python-api-connection)
- [WebSocket Fallback Setup](#websocket-fallback-setup-optional)
- [Integration Checklist](#integration-checklist)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware
- PC with Unreal Engine 5.0+ installed
- 6GB+ free disk space for UE5 project
- GPU recommended (NVIDIA RTX or AMD equivalent)

### Software
- Unreal Engine 5.0, 5.1, 5.2, or 5.3
- Python 3.9 - 3.11 (matching UE5's bundled Python version)
- Check UE5 Python version:
  ```
  Editor Preferences → Python → Show Console on Startup
  # Console will display Python version
  ```

### Network
- Python and UE5 on same machine (default)
- If remote: ensure port 8765 (WebSocket) is open

---

## Enable Python Plugin

### Step 1: Open UE5 Editor

Launch any UE5 project (or create new Third-Person template project).

### Step 2: Access Plugin Manager

```
UE5 Menu → Edit → Plugins
```

### Step 3: Search and Enable Python Plugin

1. Search bar: type "Python"
2. Find "Python Editor Script Plugin"
3. Click checkbox to enable
4. Restart editor when prompted

**After restart, you should see:**
- Python console window (if "Show Console on Startup" enabled)
- `[Python] Python Engine Initialized` in Output Log
- New menu: `Tools → Execute Python Script`

### Step 4: Enable Python REPL (Optional)

For interactive debugging:
```
Edit → Preferences → Python → Python REPL
# Restart editor
```

Now you can use Python console with `>>>` prompt.

---

## Create Hand Skeleton Asset

### Option 1: Import from Marketplace (Easiest)

UE5 has free hand skeletal mesh assets in the Epic Games content library:

1. Open Content Browser
2. View Options → Show Plugin Content
3. Click "Add/Import"
4. Search "Digital Human"
5. Download and import hand skeleton mesh

This includes pre-rigged 21-bone skeleton (MetaHuman hands).

### Option 2: Create Custom Hand Mesh

If you need a custom hand:

1. **Export from Blender:**
   - Model hand with 21 bones matching `LANDMARK_TO_BONE_NAME`
   - Bone naming: `hand_wrist`, `hand_thumb_01-04`, `hand_index_01-04`, etc.
   - Export as FBX with skeletal mesh

2. **Import into UE5:**
   - Content Browser → Import
   - Select FBX file
   - Enable "Skeletal Mesh"
   - Skeleton: assign or create new
   - Make sure bones retain their names exactly

### Bone Naming Reference

All 21 bones must be named exactly as shown:

```
0:  hand_wrist           # Base / wrist
1:  hand_thumb_01        # Thumb proximal
2:  hand_thumb_02        # Thumb intermediate
3:  hand_thumb_03        # Thumb distal
4:  hand_thumb_04        # Thumb tip
5:  hand_index_01        # Index proximal
6:  hand_index_02        # Index intermediate
7:  hand_index_03        # Index distal
8:  hand_index_04        # Index tip
9:  hand_middle_01       # Middle proximal
10: hand_middle_02       # Middle intermediate
11: hand_middle_03       # Middle distal
12: hand_middle_04       # Middle tip
13: hand_ring_01         # Ring proximal
14: hand_ring_02         # Ring intermediate
15: hand_ring_03         # Ring distal
16: hand_ring_04         # Ring tip
17: hand_pinky_01        # Pinky proximal
18: hand_pinky_02        # Pinky intermediate
19: hand_pinky_03        # Pinky distal
20: hand_pinky_04        # Pinky tip
```

**Verify bone names in UE5:**
```
Content Browser → Double-click skeleton → Skeleton Tree panel
# Check all bone names exactly match the list above
```

---

## Setup Hand Animation Blueprint

### Step 1: Create Animation Blueprint

1. Right-click in Content Browser
2. Create → Blueprint → Blueprint Class
3. Choose parent: "AnimInstance"
4. Name: `HandAnimBP_Left` (and `HandAnimBP_Right`)

### Step 2: Open Animation Graph

1. Double-click `HandAnimBP_Left`
2. Click "Anim Graph" tab (bottom)

### Step 3: Add IK Bone Transform

In the Anim Graph editor:

1. Right-click → Search "Skeletal Control" → "Two Bone IK"
2. Wire structure: Input Pose → Two Bone IK → Output Pose
3. Configure Two Bone IK:
   - IK Bone: `hand_middle_03`
   - Lower Limb Bone: `hand_middle_02`
   - Upper Limb Bone: `hand_middle_01`
4. Set IK Location (from external input, we'll set from Python)

### Step 4: Add Event Graph for Python Callbacks

Click "Event Graph" tab:

```blueprint
# Create these functions (right-click → New Function)

Function: UpdateBoneTransform
  Inputs: BoneName (String), Location (Vector), Rotation (Rotator)
  Implementation:
    - Get Skeletal Mesh Component
    - Set Bone Transform with SetBoneTransformByName
```

Example (pseudocode):
```cpp
void AHandCharacter::UpdateBoneTransform(FString BoneName, FVector Loc, FRotator Rot)
{
    if (SkeletalMeshComponent)
    {
        FTransform NewTransform;
        NewTransform.SetLocation(Loc);
        NewTransform.SetRotation(Rot.Quaternion());
        SkeletalMeshComponent->SetBoneTransformByName(FName(*BoneName), NewTransform, EBoneSpaces::ComponentSpace);
    }
}
```

---

## Python API Connection

### Step 1: Verify Python API Available

Run in editor's Python console:
```python
import unreal
print(f"UE5 Python API version: {unreal.VERSION}")
# Output: UE5 Python API version: (5, 1, 0)
```

### Step 2: Get Editor World Reference

```python
# In Python console (Tools → Execute Python Script):
import unreal

# Get editor world
editor_world = unreal.get_editor_world()
print(f"Editor world: {editor_world}")

# Get all skeletal mesh actors
skeletal_actors = unreal.GameplayStatics.get_all_actors_of_class(
    editor_world,
    unreal.SkeletalMeshActor
)
print(f"Found {len(skeletal_actors)} skeletal mesh actors")
```

### Step 3: Connect via UnrealPythonAPIBridge

```python
from unreal_bridge import UnrealPythonAPIBridge

bridge = UnrealPythonAPIBridge("/path/to/ue5/project")

if bridge.connect():
    print("Connected to UE5!")

    # Send test update
    from unreal_bridge import SkeletalMeshUpdate, BoneTransform

    test_update = SkeletalMeshUpdate(
        hand_id=0,
        hand_side="Left",
        bone_transforms=[
            BoneTransform("hand_wrist", (0, 0, 0), (0, 0, 0)),
            BoneTransform("hand_index_01", (0.1, 0, 0), (0, 0, 0)),
        ]
    )

    success = bridge.send_skeletal_update(test_update)
    print(f"Update sent: {success}")
else:
    print("Connection failed!")
```

---

## WebSocket Fallback Setup (Optional)

If Python API is unavailable (e.g., engine not running), use WebSocket:

### Step 1: Enable WebSocket in Config

```python
# In config.py
unreal_use_websocket = True
unreal_websocket_endpoint = "ws://localhost:8765"
```

### Step 2: Create Unreal Plugin Receiver

You would need a custom UE5 plugin to receive WebSocket messages. Example structure:

```cpp
// MyWebSocketPlugin.cpp
#include "Networking.h"
#include "WebSocketsModule.h"

void FMyWebSocketModule::StartListening()
{
    WebSocket = FWebSocketsModule::Get().CreateWebSocket("ws://localhost:8765");
    WebSocket->OnMessage().AddRaw(this, &FMyWebSocketModule::OnMessage);
    WebSocket->Connect();
}

void FMyWebSocketModule::OnMessage(const FString& Msg)
{
    // Parse JSON: {"type": "skeleton_update", "bones": [...]}
    // Update skeletal mesh bones
}
```

**Note:** This is advanced. For most cases, use Python API directly.

---

## Integration Checklist

Before running `example_unreal_integration.py`:

- [ ] UE5 5.0+ installed
- [ ] Python plugin enabled and editor restarted
- [ ] Hand skeletal mesh created/imported with 21 bones
- [ ] All bone names verified (exactly match list above)
- [ ] Animation Blueprint created
- [ ] IK bone transform setup
- [ ] Python console shows no errors
- [ ] `unreal` module imports successfully
- [ ] Project file is saved (File → Save)
- [ ] Editor is running (Python API requires active editor instance)

### Run Test

```bash
# Terminal (not in UE5 Python console)
python example_unreal_integration.py
```

**If successful:**
```
========================================
EXAMPLE 3: UNREAL ENGINE 5 INTEGRATION
========================================

[Initialization] Setting up Unreal Engine bridge...
  Python API connection successful
[Initialization] Creating hand skeleton converter...

[Running] Simulating UE5 skeletal animation...
  Duration: 10.0s (300 frames)
  Hand count: 2 (Left + Right)
  Bones per hand: 21
  Gesture sequence: reach_forward → wave → grab_and_release → reach_forward

Frame   0: Gesture=reach_forward    Updates= 42 Latency=  1.23ms Errors=  0
Frame  60: Gesture=wave             Updates=102 Latency=  1.45ms Errors=  0
...

[Results Summary]
========================================
Total frames processed: 300
Average frame latency: 1.34ms
FPS achieved: 746.3

[Skeletal Animation Performance]
  Total skeletal updates sent: 599
  Expected updates: 600 (2 hands per frame)
  Update success rate: 99.8%
```

---

## Troubleshooting

### "Python API not available"

**Cause:** Python plugin not enabled or editor not running

**Fix:**
1. Edit → Plugins → search "Python" → enable
2. Restart editor
3. Verify: Tools menu has "Execute Python Script" option
4. Try: `python example_unreal_integration.py` with editor running

### "Bones not updating in viewport"

**Cause:** Bone names don't match, IK not configured, or animation blueprint not assigned

**Fix:**
1. Verify bone names exactly match LANDMARK_TO_BONE_NAME
   ```
   Content Browser → Skeleton → Skeleton Tree
   # Check each bone name character-by-character
   ```
2. Verify Animation Blueprint assigned to skeletal mesh
3. In Animation BP: Check "Two Bone IK" is wired correctly
4. Test manual bone update in Python console

### "Latency very high (>500ms)"

**Cause:** Editor running slowly, complex scene, or network congestion

**Fix:**
- Close other applications
- Reduce scene complexity
- Check GPU utilization (Window → Stat → GPU)
- If remote: reduce UE5 scene detail

### "WebSocket connection refused"

**Cause:** Plugin not installed or port 8765 blocked

**Fix:**
- This is expected without WebSocket plugin
- Use Python API instead (easier)
- For WebSocket: install custom plugin from UE5 marketplace

### "Connection succeeds but no bone updates visible"

**Cause:** Skeletal mesh component not found, or wrong actor

**Fix:**
```python
# Debug: list all skeletal mesh actors
import unreal
world = unreal.get_editor_world()
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.SkeletalMeshActor)
for actor in actors:
    print(f"Actor: {actor.get_name()} → {actor.skeletal_mesh_component}")
```

---

## Performance Tips

| Setting | Impact | Recommendation |
|---------|--------|-----------------|
| Update rate | Frequency of bone updates | 30-60 Hz typical |
| Bone count | Render cost | 21 bones per hand: acceptable |
| Animation blueprint complexity | CPU time | Keep simple (IK + FK only) |
| Viewport resolution | GPU time | 1920x1080 default |

**For real-time performance:**
- Target: 10-20ms per frame with Python API
- With WebSocket: add 5-10ms network latency
- Total: <50ms roundtrip gesture → UE5 animation

---

## Integration with Gesture System

Once working with example script, integrate into main pipeline:

```python
# In main_3d_vr.py
if config.enable_ue5_integration:
    from unreal_bridge import UnrealPythonAPIBridge

    bridge = UnrealPythonAPIBridge(config.unreal_project_path)
    bridge.connect()

    # In main loop:
    for hand_id in [0, 1]:
        hand_landmarks_3d = hands_3d[hand_id].landmarks_3d
        transforms = converter.landmarks_to_bone_transforms(
            hand_landmarks_3d,
            "Left" if hand_id == 0 else "Right"
        )
        update = SkeletalMeshUpdate(hand_id, "Left" if hand_id == 0 else "Right", transforms)
        bridge.send_skeletal_update(update)
```

---

## Summary

1. **Setup:** Enable Python plugin, create skeletal mesh
2. **Configure:** Name all 21 bones correctly
3. **Test:** Run example script with editor open
4. **Integrate:** Add UE5 pipeline to main gesture system
5. **Performance:** Typical latency <50ms, achieves 20+ FPS with full system

Real-time hand skeleton animation in UE5 powered by MediaPipe hand detection + gesture recognition.
