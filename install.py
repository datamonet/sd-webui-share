import launch

if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface-hub==0.13.4", "requirements for WebUI Share")


if not launch.is_installed("send2trash"):
    launch.run_pip("install Send2Trash", "requirement for WebUI Share")
