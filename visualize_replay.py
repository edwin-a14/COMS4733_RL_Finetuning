"""Visualize what happens during demo replay."""
import numpy as np
import json
from env.mujoco_env import FrankaPickPlaceEnv

env = FrankaPickPlaceEnv(asset_root="./env/mujoco_assets")

# Test episode_0000 from new dataset
episode = "dataset/episode_0000"
actions = np.load(f"{episode}/actions.npy")
with open(f"{episode}/meta.json") as f:
    meta = json.load(f)

print(f"Testing {episode}")
print(f"Target: {meta['target_color']}, Length: {meta['episode_length']}")

obs, info = env.reset(hindered=False)

# Get target object body ID
target = info['target_color']
target_body_id = env.model.body(f'object_body_{target}').id
bin_body_id = env.model.body('bin').id

print(f"\nReplaying {len(actions)} actions...")
print(f"{'Step':>5} {'ObjZ':>6} {'BinZ':>6} {'Gripper':>7} {'Reward':>7} {'Success':>7}")
print("-" * 50)

for step in range(len(actions)):
    action = actions[step]
    step_result = env.step(action)
    
    # Get positions
    obj_pos = env.data.xpos[target_body_id]
    bin_pos = env.data.xpos[bin_body_id]
    gripper = action[-1]  # Last dimension is gripper
    
    success = step_result.info.get("success", False)
    
    if step % 20 == 0 or success:
        print(f"{step:5d} {obj_pos[2]:6.3f} {bin_pos[2]:6.3f} {gripper:7.3f} {step_result.reward:7.3f} {int(success):7d}")
    
    if step_result.terminated or step_result.truncated:
        break

# Check final state
obj_pos = env.data.xpos[target_body_id]
bin_pos = env.data.xpos[bin_body_id]
horiz_dist = np.sqrt((obj_pos[0] - bin_pos[0])**2 + (obj_pos[1] - bin_pos[1])**2)
height_diff = obj_pos[2] - bin_pos[2]

print(f"\nFinal state:")
print(f"  Object position: {obj_pos}")
print(f"  Bin position: {bin_pos}")
print(f"  Horizontal distance: {horiz_dist:.4f} (threshold: 0.12)")
print(f"  Height difference: {height_diff:.4f} (threshold: 0.08)")
print(f"  Would succeed: {horiz_dist < 0.12 and height_diff < 0.08}")

# Try settling
print("\nLetting physics settle for 30 steps...")
for i in range(30):
    step_result = env.step(np.zeros(8))
    success = step_result.info.get("success", False)
    
    if i % 5 == 0 or success:
        obj_pos = env.data.xpos[target_body_id]
        horiz_dist = np.sqrt((obj_pos[0] - bin_pos[0])**2 + (obj_pos[1] - bin_pos[1])**2)
        height_diff = obj_pos[2] - bin_pos[2]
        print(f"  Settle {i:2d}: obj_z={obj_pos[2]:.3f}, h_dist={horiz_dist:.4f}, h_diff={height_diff:.4f}, success={int(success)}")
    
    if success:
        print(f"  ✓ SUCCESS at settling step {i}!")
        break

env.close()
