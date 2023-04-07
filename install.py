import launch

launch.run_pip("install huggingface-hub==0.11.0", "requirements for WebUI Share")

if not launch.is_installed("send2trash"):
    launch.run_pip("install Send2Trash", "requirement for WebUI Share")
