# Copy to rig.local.sh (git-ignored) and set your rig's values; env.sh sources it.
export SQUINT_PORT=/dev/ttyACM0                 # macOS: /dev/tty.usbmodem...   (lerobot-find-port)
export SQUINT_CAMERA=/dev/video0                # macOS: an integer index, e.g. 0
export SQUINT_CAMERA_ROTATION=0                 # 0, 90, 180 or -90, if your wrist cam is mounted rotated
export SQUINT_ROBOT_ID=so101_follower_arm       # your calibration file name, without .json
export SQUINT_CALIBRATION_DIR=deploy_utils      # folder holding that file
