@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem 이 PC 의 서버(기본 8000)를 Cloudflare 임시 터널로 연다. 서버(start_server.bat)는 따로 켜 둔다.
if not defined PORT set "PORT=8000"

set "CLOUDFLARED="
for %%P in (cloudflared.exe) do set "CLOUDFLARED=%%~$PATH:P"
if not defined CLOUDFLARED if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "CLOUDFLARED=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not defined CLOUDFLARED if exist "%ProgramFiles%\cloudflared\cloudflared.exe" set "CLOUDFLARED=%ProgramFiles%\cloudflared\cloudflared.exe"
if not defined CLOUDFLARED (
    echo [오류] cloudflared 를 찾지 못했습니다. 설치: winget install Cloudflare.cloudflared
    pause
    exit /b 1
)

echo [FindUs] Cloudflare 터널을 엽니다: http://127.0.0.1:%PORT%
echo 아래 로그의 https://....trycloudflare.com 주소로 다른 PC 에서 접속합니다 (켤 때마다 주소가 바뀜).
echo 다른 PC 에서 쓰려면 .env 에 API_KEY 와 WORKBENCH_REMOTE=true 가 있어야 합니다 (README "다른 PC 에서 쓰기").
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo.

"%CLOUDFLARED%" tunnel --url http://127.0.0.1:%PORT%
set "TUNNEL_EXIT_CODE=%ERRORLEVEL%"
if not "%TUNNEL_EXIT_CODE%"=="0" (
    echo.
    echo [오류] 터널이 종료되었습니다. 위의 오류 내용을 확인해 주세요.
    pause
)
exit /b %TUNNEL_EXIT_CODE%
