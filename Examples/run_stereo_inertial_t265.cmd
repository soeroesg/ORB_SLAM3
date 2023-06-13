@echo off

set EXAMPLES_DIR=%~dp0
call %EXAMPLES_DIR%\windows-run-config.cmd

%ORBSLAM3_BIN_DIR%\stereo_inertial_realsense_t265.exe ^
  %ORB_VOCABULARY% ^
  %ORBSLAM3_ROOT_DIR%/Examples/Stereo-Inertial/RealSense_T265.yaml

