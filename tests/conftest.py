import os

# map-api rebuilds a place's map whenever a walk finalizes (on by default); tests that want it pass auto_build=True
# and a stand-in build, so a finished test recording never starts a real build.
os.environ["NAV_AUTO_BUILD"] = "0"
