@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Python 경로를 직접 정한 경우 우선 사용한다.
set "SERVER_PYTHON="
if defined PYTHON_EXE (
    set "SERVER_PYTHON=%PYTHON_EXE%"
    goto run
)
if exist ".venv\Scripts\python.exe" (
    set "SERVER_PYTHON=%CD%\.venv\Scripts\python.exe"
    goto run
)
if exist "..\ai_lab\.venv\Scripts\python.exe" (
    set "SERVER_PYTHON=%CD%\..\ai_lab\.venv\Scripts\python.exe"
    goto run
)
for %%P in (python.exe) do set "SERVER_PYTHON=%%~$PATH:P"
if defined SERVER_PYTHON goto run

echo [오류] 서버를 실행할 Python을 찾지 못했습니다.
echo README.md의 설치 순서대로 가상환경과 패키지를 준비해 주세요.
echo 다른 가상환경은 PYTHON_EXE에 python.exe 전체 경로를 지정하면 됩니다.
set "SERVER_EXIT_CODE=1"
goto finish

:run
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"
set "SERVER_URL_HOST=%HOST%"
if "%HOST%"=="0.0.0.0" set "SERVER_URL_HOST=127.0.0.1"

echo [FindUs] 서버를 시작합니다.
echo Python: "%SERVER_PYTHON%"
echo 나이 변환: http://%SERVER_URL_HOST%:%PORT%/test/aging
echo 후처리: http://%SERVER_URL_HOST%:%PORT%/test/post
echo API 문서: http://%SERVER_URL_HOST%:%PORT%/docs
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo.

"%SERVER_PYTHON%" -u -m app
set "SERVER_EXIT_CODE=%ERRORLEVEL%"

:finish
if not "%SERVER_EXIT_CODE%"=="0" (
    echo.
    echo [오류] 서버가 종료되었습니다. 위의 오류 내용을 확인해 주세요.
    echo 설치 안내: README.md / docs\[02]_구축_가이드.md
    pause
)
popd
exit /b %SERVER_EXIT_CODE%
