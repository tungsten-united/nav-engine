import os

# map-api rebuilds a place's map whenever a walk finalizes (on by default); tests that want it pass auto_build=True
# and a stand-in build, so a finished test recording never starts a real build.
os.environ["NAV_AUTO_BUILD"] = "0"
# A build makes a 3D model of the place (nav/mapping/geometry.py), whose GPU stage queues on the machine's shared GPU
# (gpu-run); tests build without it, from dead reckoning.
os.environ["NAV_GEOMETRY"] = "0"
# The cloud map-api keeps the inference host's feature cache warm (nav/map_api/warm.py); a test app never talks to it.
os.environ["NAV_POSE_WARM"] = "0"
