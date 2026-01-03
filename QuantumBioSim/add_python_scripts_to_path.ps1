# Add Python Scripts to PATH
# This script adds the Python user scripts directory to the system PATH permanently

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Adding Python Scripts to PATH" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptsPath = "C:\Users\mbpd1\AppData\Roaming\Python\Python313\Scripts"

# Check if the directory exists
if (-not (Test-Path $scriptsPath)) {
    Write-Host "Error: Directory does not exist: $scriptsPath" -ForegroundColor Red
    exit 1
}

# Get current PATH from User environment
$currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if already in PATH
if ($currentUserPath -like "*$scriptsPath*") {
    Write-Host "[OK] Python Scripts directory is already in PATH" -ForegroundColor Green
    Write-Host ""
    Write-Host "Current User PATH:" -ForegroundColor Yellow
    Write-Host $currentUserPath
    exit 0
}

Write-Host "Adding to User PATH: $scriptsPath" -ForegroundColor Yellow

# Add to PATH (User level, not System to avoid permission issues)
try {
    $newPath = $currentUserPath + ";" + $scriptsPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    
    Write-Host "[OK] Successfully added to User PATH!" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: You must restart your terminal for changes to take effect!" -ForegroundColor Yellow -BackgroundColor DarkRed
    Write-Host ""
    Write-Host "To apply in current session, run:" -ForegroundColor Cyan
    Write-Host '  $env:Path = [Environment]::GetEnvironmentVariable("Path", "User")' -ForegroundColor White
    Write-Host ""
    
    # Show updated PATH
    Write-Host "New User PATH:" -ForegroundColor Green
    Write-Host $newPath
    
}
catch {
    Write-Host "Error setting PATH: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PATH Update Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
