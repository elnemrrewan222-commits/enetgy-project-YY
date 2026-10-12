@echo off
title GridPulse Energy Forecaster
echo ==========================================================
echo Starting GridPulse Energy Forecasting Web Application...
echo ==========================================================
if exist "C:\Users\HP\anaconda3\python.exe" (
    "C:\Users\HP\anaconda3\python.exe" ui.py
) else if exist "C:\Users\HP\anaconda3\envs\myenv\python.exe" (
    "C:\Users\HP\anaconda3\envs\myenv\python.exe" ui.py
) else (
    python ui.py
)
pause
