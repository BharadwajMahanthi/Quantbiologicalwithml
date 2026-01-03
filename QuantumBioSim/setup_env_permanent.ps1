# Permanent Environment Setup for OpenMC
# Run this once to set system-wide environment variables

Write-Host "Setting Permanent Environment Variables for OpenMC..." -ForegroundColor Cyan

# 1. Add OpenMC executable to PATH (User level)
$openMCPath = "C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\openmc\build\bin\Release"
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -notlike "*$openMCPath*") {
    $newPath = "$currentPath;$openMCPath"
    [System.Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "SUCCESS: Added OpenMC to PATH" -ForegroundColor Green
    Write-Host "  Path: $openMCPath" -ForegroundColor Gray
}
else {
    Write-Host "ALREADY SET: OpenMC already in PATH" -ForegroundColor Yellow
}

# 2. Set OPENMC_CROSS_SECTIONS (User level)
$crossSectionsPath = "C:\Users\mbpd1\.openmc\endfb-vii.1-hdf5\cross_sections.xml"
[System.Environment]::SetEnvironmentVariable("OPENMC_CROSS_SECTIONS", $crossSectionsPath, "User")
Write-Host "SUCCESS: Set OPENMC_CROSS_SECTIONS" -ForegroundColor Green
Write-Host "  Path: $crossSectionsPath" -ForegroundColor Gray

# 3. Verify
Write-Host "`nVerifying..." -ForegroundColor Cyan
$verifyPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$verifyXS = [System.Environment]::GetEnvironmentVariable("OPENMC_CROSS_SECTIONS", "User")

Write-Host "  PATH contains openmc:" ($verifyPath -like "*openmc*") -ForegroundColor Gray
Write-Host "  OPENMC_CROSS_SECTIONS:" $verifyXS -ForegroundColor Gray

Write-Host "`nSUCCESS: Environment variables set permanently!" -ForegroundColor Green
Write-Host "`nIMPORTANT: Close ALL terminals and open a new one for changes to take effect." -ForegroundColor Yellow
Write-Host "Then verify with:" -ForegroundColor Cyan
Write-Host '  openmc --version' -ForegroundColor Gray
Write-Host '  python -c "import openmc; print(openmc.config[''cross_sections''])"' -ForegroundColor Gray
