param(
  [switch]$Build
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$tauriConfigPath = Join-Path $root "src-tauri\tauri.conf.json"
$cargoTomlPath = Join-Path $root "src-tauri\Cargo.toml"
$targetRelease = Join-Path $root "src-tauri\target\release"
$bundleNsis = Join-Path $targetRelease "bundle\nsis"
$portableDir = Join-Path $root "release\portable"
$installerDir = Join-Path $root "release\installer"

function Get-TomlValue {
  param(
    [string]$Path,
    [string]$Key
  )

  $line = Get-Content -Path $Path | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
  if (-not $line) {
    throw "Could not read $Key from $Path"
  }
  return ($line -replace "^\s*$Key\s*=\s*`"", "" -replace "`"\s*$", "")
}

if ($Build) {
  $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
  if (Test-Path $cargoBin) {
    $env:Path = "$cargoBin;$env:Path"
  }

  Push-Location $root
  try {
    npm run tauri:build
  } finally {
    Pop-Location
  }
}

$tauriConfig = Get-Content -Path $tauriConfigPath -Raw | ConvertFrom-Json
$version = [string]$tauriConfig.version
$packageName = Get-TomlValue -Path $cargoTomlPath -Key "name"
$portableSource = Join-Path $targetRelease "$packageName.exe"
$portableFileName = Split-Path -Leaf $portableSource

if (-not (Test-Path $portableSource)) {
  throw "Portable executable was not found: $portableSource. Run npm run release:build first."
}

$installerSource = Get-ChildItem -Path $bundleNsis -Filter "*$version*_x64-setup.exe" -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $installerSource) {
  throw "Installer executable was not found in $bundleNsis. Run npm run release:build first."
}

New-Item -ItemType Directory -Force -Path $portableDir, $installerDir | Out-Null
Get-ChildItem -Path $portableDir -Filter "*.exe" -File -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path $installerDir -Filter "*.exe" -File -ErrorAction SilentlyContinue | Remove-Item -Force

Copy-Item -Path $portableSource -Destination (Join-Path $portableDir $portableFileName) -Force
Copy-Item -Path $installerSource.FullName -Destination (Join-Path $installerDir $installerSource.Name) -Force

Write-Host "Release files published:"
Write-Host "  Portable:  $(Join-Path $portableDir $portableFileName)"
Write-Host "  Installer: $(Join-Path $installerDir $installerSource.Name)"
