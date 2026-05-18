# GitHub上传脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  毫米波雷达摔倒检测系统 - GitHub上传" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 刷新PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 检查Git
Write-Host "[1/7] 检查Git安装..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "  ✓ Git已安装: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Git未安装，请先安装Git" -ForegroundColor Red
    exit 1
}

# 进入项目目录
Set-Location "d:\Trae\fall-detection\radar_fall_detection_demo"

# 初始化Git仓库
Write-Host "[2/7] 初始化Git仓库..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
    Write-Host "  ✓ Git仓库已初始化" -ForegroundColor Green
} else {
    Write-Host "  ✓ Git仓库已存在" -ForegroundColor Green
}

# 配置用户信息（请替换为你的GitHub信息）
Write-Host "[3/7] 配置Git用户信息..." -ForegroundColor Yellow
$userName = "Best6good"
$userEmail = "best6good@github.com"
git config user.name $userName
git config user.email $userEmail
Write-Host "  ✓ 用户名: $userName" -ForegroundColor Green
Write-Host "  ✓ 邮箱: $userEmail" -ForegroundColor Green

# 创建.gitignore文件
Write-Host "[4/7] 创建.gitignore文件..." -ForegroundColor Yellow
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs and Alarms
logs/
alarms/
*.log

# OS
.DS_Store
Thumbs.db
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent -Encoding UTF8
Write-Host "  ✓ .gitignore文件已创建" -ForegroundColor Green

# 添加文件到暂存区
Write-Host "[5/7] 添加文件到暂存区..." -ForegroundColor Yellow
git add .
Write-Host "  ✓ 所有文件已添加到暂存区" -ForegroundColor Green

# 提交
Write-Host "[6/7] 提交更改..." -ForegroundColor Yellow
$commitMessage = "feat: 毫米波雷达摔倒检测系统 v2.1.1

- 新增骨架可视化功能（基于点云处理识别）
- 新增宠物识别过滤功能
- 优化行走与站立状态差异化
- 优化点云生成算法（毫米波雷达特性）
- 优化摔倒检测算法（多特征融合）"
git commit -m $commitMessage
Write-Host "  ✓ 更改已提交" -ForegroundColor Green

# 添加远程仓库并推送
Write-Host "[7/7] 推送到GitHub..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/Best6good/fall_detection.git"

# 移除已存在的origin（如果有）
git remote remove origin 2>$null

# 添加远程仓库
git remote add origin $remoteUrl
Write-Host "  ✓ 远程仓库已添加: $remoteUrl" -ForegroundColor Green

# 推送到GitHub
Write-Host "  正在推送到GitHub..." -ForegroundColor Cyan
git branch -M main
git push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ✓ 上传成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "项目地址: https://github.com/Best6good/fall_detection" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  ✗ 上传失败！" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "请检查:" -ForegroundColor Yellow
    Write-Host "1. GitHub仓库是否存在" -ForegroundColor Yellow
    Write-Host "2. 网络连接是否正常" -ForegroundColor Yellow
    Write-Host "3. GitHub凭据是否配置正确" -ForegroundColor Yellow
}

Write-Host ""
