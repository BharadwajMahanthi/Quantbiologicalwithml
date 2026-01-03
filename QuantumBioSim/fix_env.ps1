# Fix OpenMC Cross Sections Environment Variable
# This script ensures the environment variable is set correctly for THIS session

Write-Host "Checking OpenMC Cross Sections Configuration..." -ForegroundColor Cyan

# 1. Check current process environment
$processEnv = $env:OPENMC_CROSS_SECTIONS
Write-Host "`nCurrent Process Environment:" -ForegroundColor Yellow
Write-Host "  OPENMC_CROSS_SECTIONS = $processEnv"

# 2. Check user registry
$correctPath = "C:\Users\mbpd1\.openmc\endfb-vii.1-hdf5\cross_sections.xml"
$userEnv = [System.Environment]::GetEnvironmentVariable("OPENMC_CROSS_SECTIONS", "User")
Write-Host "`nUser Registry Value:" -ForegroundColor Yellow
Write-Host "  OPENMC_CROSS_SECTIONS = $userEnv"

# 3. Set it for THIS session immediately
Write-Host "`nSetting for current session..." -ForegroundColor Cyan
$env:OPENMC_CROSS_SECTIONS = $correctPath

# 4. Verify
Write-Host "`nVerifying in Python..." -ForegroundColor Cyan
python -c "import os; print('  Process env:', os.environ.get('OPENMC_CROSS_SECTIONS', 'NOT SET'))"

# 5. Test OpenMC import (should be clean now)
Write-Host "`nTesting OpenMC import..." -ForegroundColor Cyan
python -c "import openmc; print('  OpenMC config:', openmc.config['cross_sections'])"

Write-Host "`nSUCCESS! Environment variable is now set for this session." -ForegroundColor Green
Write-Host "For NEW terminal sessions, it will automatically use: $correctPath" -ForegroundColor Green
