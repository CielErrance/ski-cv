# 在 cv 目录运行。需先完成一次 GitHub 登录：
#   gh auth login
# 若 HTTPS 连不上 github.com，可先开代理，或把 SSH 公钥加到 GitHub 后用 SSH 推送。

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 gh，请先安装 GitHub CLI: winget install GitHub.cli"
}

$auth = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "请先登录 GitHub："
    gh auth login --hostname github.com --git-protocol https --web
}

$repo = "CielErrance/ski-cv"
$exists = gh repo view $repo 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "创建远程仓库 $repo ..."
    gh repo create ski-cv --public --source=. --remote=origin --push
} else {
    Write-Host "仓库已存在，推送到 main ..."
    git branch -M main
    git remote set-url origin "https://github.com/$repo.git"
    git push -u origin main
}

Write-Host "完成: https://github.com/$repo"
