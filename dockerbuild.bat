@echo off
setlocal enabledelayedexpansion

set IMAGE_REPO=fuckdocker42731/twincle
set DOCKERFILE=benchmarks\Dockerfile.bench

echo Building and pushing Docker images...

call :build_and_push 12.4.1 %IMAGE_REPO%:cuda12.4
if errorlevel 1 exit /b 1

call :build_and_push 12.2.2 %IMAGE_REPO%:cuda12.2
if errorlevel 1 exit /b 1

call :build_and_push 11.8.0 %IMAGE_REPO%:cuda11.8
if errorlevel 1 exit /b 1

echo Done.
exit /b 0

:build_and_push
set CUDA_TAG=%~1
set IMAGE=%~2
echo.
echo === %IMAGE% (CUDA %CUDA_TAG%) ===
docker build -f %DOCKERFILE% --build-arg CUDA_TAG=%CUDA_TAG% -t %IMAGE% .
if errorlevel 1 exit /b 1
docker push %IMAGE%
if errorlevel 1 exit /b 1
exit /b 0