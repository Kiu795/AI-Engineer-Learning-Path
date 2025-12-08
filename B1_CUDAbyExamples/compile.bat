@echo off
call "D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc %1 -o %~n1.exe
%~n1.exe