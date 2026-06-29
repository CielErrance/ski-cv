# Run from the cv directory. First-time setup: gh auth login

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) not found. Install: winget install GitHub.cli"
}

gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    gh auth login --hostname github.com --git-protocol https --web
}

$repo = "CielErrance/ski-cv"
gh repo view $repo 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gh repo create ski-cv --public --source=. --remote=origin --push
} else {
    git branch -M main
    git remote set-url origin "https://github.com/$repo.git"
    git push -u origin main
}

Write-Host "Done: https://github.com/$repo"
