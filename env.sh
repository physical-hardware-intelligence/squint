# Source from the repo root on macOS, every session:  source env.sh
# Vulkan on macOS through MoltenVK (Vulkan -> Metal). Download MoltenVK-macos.tar v1.4.2 into ./MoltenVK
# (see docs/so101-phi.md).
export VK_ICD_FILENAMES="$PWD/MoltenVK/MoltenVK/dynamic/dylib/macOS/MoltenVK_icd.json"
export MVK_CONFIG_LOG_LEVEL=1   # MoltenVK: errors only (otherwise ~300 lines per start)
# This machine's rig: SQUINT_* port, camera, calibration (git-ignored; template in rig.example.sh)
if [ -f rig.local.sh ]; then . ./rig.local.sh; fi
