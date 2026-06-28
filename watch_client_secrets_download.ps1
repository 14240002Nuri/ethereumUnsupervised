param(
    [string]$DownloadDir = "$env:USERPROFILE\Downloads",
    [string]$TargetFile = "c:\Users\DIT\OneDrive - idsMED\Bismillah Tesis\Apps\client_secrets.json",
    [int]$TimeoutMinutes = 20
)

$ErrorActionPreference = "Stop"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

Write-Host "Watching for OAuth JSON download in: $DownloadDir"
Write-Host "Will copy to: $TargetFile"
Write-Host "Timeout: $TimeoutMinutes minutes"

function Test-OAuthJson($path) {
    try {
        $raw = Get-Content -Raw -LiteralPath $path
        $json = $raw | ConvertFrom-Json
        if ($null -ne $json.installed.client_id -and $null -ne $json.installed.client_secret) {
            return $true
        }
        if ($null -ne $json.web.client_id -and $null -ne $json.web.client_secret) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

while ((Get-Date) -lt $deadline) {
    $candidates = Get-ChildItem -LiteralPath $DownloadDir -Filter *.json -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    foreach ($file in $candidates) {
        if (Test-OAuthJson $file.FullName) {
            Copy-Item -LiteralPath $file.FullName -Destination $TargetFile -Force
            Write-Host "Copied OAuth client file:"
            Write-Host "Source: $($file.FullName)"
            Write-Host "Target: $TargetFile"
            exit 0
        }
    }

    Start-Sleep -Seconds 3
}

Write-Error "Timed out waiting for OAuth client JSON in $DownloadDir"
