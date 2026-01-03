# OpenMC Environment Setup Script (PowerShell)
# Bypasses PATH length limitations using Registry methods
# Run as: .\setup_openmc_env.ps1

Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "OpenMC Environment Variable Setup" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

# Define paths
$CrossSectionsPath = "C:\Users\mbpd1\.openmc\endfb-viii.0\cross_sections.xml"
$OpenMCBinPath = "C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\openmc\build\bin\Release"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`nWARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "Setting variables for current user only (recommended)" -ForegroundColor Yellow
    $scope = "User"
}
else {
    Write-Host "`nRunning as Administrator" -ForegroundColor Green
    Write-Host "Setting variables system-wide" -ForegroundColor Green
    $scope = "Machine"
}

# Registry path based on scope
if ($scope -eq "User") {
    $regPath = "HKCU:\Environment"
}
else {
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
}

Write-Host "`nUsing registry path: $regPath" -ForegroundColor Gray

# ============================================================
# 1. Set OPENMC_CROSS_SECTIONS
# ============================================================
Write-Host "`n[1/2] Setting OPENMC_CROSS_SECTIONS..." -ForegroundColor Cyan

try {
    # Check if file exists
    if (Test-Path $CrossSectionsPath) {
        # Set via Registry (bypasses length limits)
        Set-ItemProperty -Path $regPath -Name "OPENMC_CROSS_SECTIONS" -Value $CrossSectionsPath -Type String
        
        # Also set for current session
        $env:OPENMC_CROSS_SECTIONS = $CrossSectionsPath
        
        Write-Host "  SUCCESS: OPENMC_CROSS_SECTIONS set to:" -ForegroundColor Green
        Write-Host "  $CrossSectionsPath" -ForegroundColor Gray
    }
    else {
        Write-Host "  WARNING: Cross sections file not found at:" -ForegroundColor Yellow
        Write-Host "  $CrossSectionsPath" -ForegroundColor Gray
        Write-Host "  Run 'python download_nuclear_data.py' first" -ForegroundColor Yellow
        
        # Set anyway for future use
        Set-ItemProperty -Path $regPath -Name "OPENMC_CROSS_SECTIONS" -Value $CrossSectionsPath -Type String
        Write-Host "  Variable set (will be active after download)" -ForegroundColor Green
    }
}
catch {
    Write-Host "  ERROR setting OPENMC_CROSS_SECTIONS: $_" -ForegroundColor Red
}

# ============================================================
# 2. Add OpenMC bin to PATH (if exists)
# ============================================================
Write-Host "`n[2/2] Adding OpenMC executable to PATH..." -ForegroundColor Cyan

try {
    if (Test-Path $OpenMCBinPath) {
        # Get current PATH from registry
        $currentPath = (Get-ItemProperty -Path $regPath -Name "Path").Path
        
        # Check if already in PATH
        if ($currentPath -notlike "*$OpenMCBinPath*") {
            # Add to PATH using registry
            $newPath = "$currentPath;$OpenMCBinPath"
            
            # Check PATH length
            if ($newPath.Length -gt 2047) {
                Write-Host "  WARNING: PATH is very long ($($newPath.Length) chars)" -ForegroundColor Yellow
                Write-Host "  Windows limit is 2047 characters" -ForegroundColor Yellow
                Write-Host "  Forcing addition anyway..." -ForegroundColor Yellow
            }
            
            # Force set (even if long)
            Set-ItemProperty -Path $regPath -Name "Path" -Value $newPath -Type ExpandString
            
            # Also set for current session
            $env:Path += ";$OpenMCBinPath"
            
            Write-Host "  SUCCESS: Added to PATH:" -ForegroundColor Green
            Write-Host "  $OpenMCBinPath" -ForegroundColor Gray
            Write-Host "  WARNING: Restart terminal for PATH changes to take effect" -ForegroundColor Yellow
        }
        else {
            Write-Host "  Already in PATH" -ForegroundColor Green
        }
    }
    else {
        Write-Host "  WARNING: OpenMC bin directory not found at:" -ForegroundColor Yellow
        Write-Host "  $OpenMCBinPath" -ForegroundColor Gray
        Write-Host "  Skipping PATH addition" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ERROR adding to PATH: $_" -ForegroundColor Red
    Write-Host "  You may need to add manually or clean up existing PATH" -ForegroundColor Yellow
}

# ============================================================
# Summary
# ============================================================
Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan

Write-Host "`nEnvironment Variables Set:" -ForegroundColor Cyan
Write-Host "  OPENMC_CROSS_SECTIONS = $CrossSectionsPath" -ForegroundColor Gray

if (Test-Path $OpenMCBinPath) {
    Write-Host "  PATH += $OpenMCBinPath" -ForegroundColor Gray
}

Write-Host "`nIMPORTANT: Close and reopen your terminal!" -ForegroundColor Yellow
Write-Host "`nVerify with:" -ForegroundColor Cyan
Write-Host '  python -c "import openmc; print(openmc.config[''cross_sections''])"' -ForegroundColor Gray
Write-Host '  openmc --version' -ForegroundColor Gray

Write-Host ""
