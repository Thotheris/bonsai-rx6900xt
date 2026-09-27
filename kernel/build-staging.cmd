@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "SOURCE=%ROOT%\_work\llama.cpp"
set "BUILD=%ROOT%\_work\build-gfx1030"
set "PATCH=%ROOT%\kernel\patches\0001-rdna2-ptq1-hip-mmvq.patch"
set "PATCH2=%ROOT%\kernel\patches\0002-rdna2-fa-kv-prefetch.patch"
set "BASE=23d0d71502d690230131f22b1393ac41d6b4406e"
if not defined BONSAI_UPSTREAM_URL set "BONSAI_UPSTREAM_URL=https://github.com/PrismML-Eng/llama.cpp.git"
if not defined BONSAI_ROCM_VENV (
    echo Set BONSAI_ROCM_VENV to the Windows Python environment containing _rocm_sdk_devel.
    exit /b 2
)
if not defined VCToolsInstallDir (
    echo Run from an x64 Visual Studio Developer Command Prompt.
    exit /b 2
)
set "SDK=%BONSAI_ROCM_VENV%\Lib\site-packages\_rocm_sdk_devel"
if not exist "%SDK%\lib\llvm\bin\clang++.exe" (
    echo Missing TheRock clang++ in %SDK%
    exit /b 2
)
if not exist "%ROOT%\_work" mkdir "%ROOT%\_work" || exit /b 2
if not exist "%SOURCE%\.git" (
    git clone --filter=blob:none --no-checkout "%BONSAI_UPSTREAM_URL%" "%SOURCE%" || exit /b 1
    git -C "%SOURCE%" fetch --depth 1 origin %BASE% || exit /b 1
    git -C "%SOURCE%" checkout --detach %BASE% || exit /b 1
)
for /f %%H in ('git -C "%SOURCE%" rev-parse HEAD') do set "ACTUAL=%%H"
if not "%ACTUAL%"=="%BASE%" (
    echo Source checkout differs from pinned PrismML revision: %ACTUAL%
    exit /b 3
)
git -C "%SOURCE%" apply --reverse --check "%PATCH%" >nul 2>&1
if errorlevel 1 (
    git -C "%SOURCE%" diff --quiet || (echo Source checkout has unrelated edits & exit /b 3)
    git -C "%SOURCE%" ls-files --others --exclude-standard | findstr . >nul && (echo Source checkout has untracked files & exit /b 3)
    git -C "%SOURCE%" apply --check "%PATCH%" || exit /b 3
    git -C "%SOURCE%" apply "%PATCH%" || exit /b 3
)
git -C "%SOURCE%" apply --reverse --check "%PATCH2%" >nul 2>&1
if errorlevel 1 (
    git -C "%SOURCE%" apply --check "%PATCH2%" || exit /b 3
    git -C "%SOURCE%" apply "%PATCH2%" || exit /b 3
)
set "PATH=%BONSAI_ROCM_VENV%\Scripts;%SDK%\bin;%SDK%\lib\llvm\bin;%PATH%"
cmake -S "%SOURCE%" -B "%BUILD%" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF || exit /b 1
cmake --build "%BUILD%" --parallel 12 --target llama-server llama-bench test-backend-ops || exit /b 1
echo Built source-patched gfx1030 binaries in %BUILD%\bin
