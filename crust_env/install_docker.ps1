Write-Host "Downloading Docker Desktop..."
Invoke-WebRequest -Uri "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" -OutFile "DockerDesktopInstaller.exe"
Write-Host "Installing Docker silently in the background... this may take 5-10 minutes."
Start-Process "DockerDesktopInstaller.exe" -ArgumentList "install --quiet --accept-license" -Wait -NoNewWindow
Write-Host "Installation initiated. Note: A system reboot is typically required after Docker Desktop installation to enable WSL2 features."
