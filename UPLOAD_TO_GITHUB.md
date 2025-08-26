# 如何将项目上传到GitHub

本指南将详细说明如何将多模态多普勒雷达人体检测项目上传到GitHub。

## 准备工作

### 1. 安装Git
```bash
# Windows (使用Git for Windows)
下载并安装: https://git-scm.com/download/win

# macOS (使用Homebrew)
brew install git

# Ubuntu/Debian
sudo apt update
sudo apt install git

# CentOS/RHEL
sudo yum install git
```

### 2. 配置Git
```bash
# 设置用户名和邮箱
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱@example.com"

# 验证配置
git config --list
```

### 3. 创建GitHub账号
- 访问 https://github.com
- 注册账号或登录现有账号

## 方法一：通过GitHub网站创建仓库

### 1. 在GitHub上创建新仓库
1. 登录GitHub
2. 点击右上角的 "+" 号
3. 选择 "New repository"
4. 填写仓库信息：
   - Repository name: `multimodal-doppler-radar-detection`
   - Description: `多模态多普勒雷达人体检测技术方案`
   - 选择 Public 或 Private
   - 不要初始化 README, .gitignore 或 LICENSE（我们已经有了）

### 2. 在本地初始化Git仓库
```bash
# 进入项目目录
cd /path/to/your/project

# 初始化Git仓库
git init

# 添加远程仓库
git remote add origin https://github.com/您的用户名/multimodal-doppler-radar-detection.git
```

### 3. 提交和推送代码
```bash
# 添加所有文件到暂存区
git add .

# 查看状态（可选）
git status

# 提交代码
git commit -m "Initial commit: 多模态多普勒雷达人体检测完整技术方案"

# 推送到GitHub
git push -u origin main
```

## 方法二：使用GitHub CLI

### 1. 安装GitHub CLI
```bash
# Windows (使用winget)
winget install --id GitHub.cli

# macOS (使用Homebrew)
brew install gh

# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### 2. 认证和创建仓库
```bash
# 登录GitHub
gh auth login

# 在项目目录下初始化并创建仓库
cd /path/to/your/project
git init
gh repo create multimodal-doppler-radar-detection --public --source=. --remote=origin --push
```

## 后续管理

### 日常提交流程
```bash
# 查看文件状态
git status

# 添加修改的文件
git add .
# 或者添加特定文件
git add src/models/new_model.py

# 提交更改
git commit -m "描述性的提交信息"

# 推送到GitHub
git push
```

### 分支管理
```bash
# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/new-feature

# 删除分支
git branch -d feature/new-feature
```

## 项目结构优化建议

### 1. 添加示例数据
由于真实数据可能很大，建议：
```bash
# 创建示例数据目录
mkdir examples/sample_data

# 添加小量示例数据用于演示
# 在.gitignore中排除大的数据文件，但保留示例
```

### 2. 文档完善
```markdown
# 在README.md中添加：
- 项目徽章 (badges)
- 安装说明
- 快速开始指南
- API文档链接
- 贡献指南
```

### 3. 发布管理
```bash
# 创建发布版本
git tag -a v1.0.0 -m "Version 1.0.0: 初始发布版本"
git push origin v1.0.0

# 在GitHub上创建Release
gh release create v1.0.0 --title "Version 1.0.0" --notes "初始发布版本"
```

## 最佳实践

### 1. 提交信息规范
```bash
# 好的提交信息示例：
git commit -m "feat: 添加多普勒谱图适配器"
git commit -m "fix: 修复CLIP模型加载问题" 
git commit -m "docs: 更新安装说明"
git commit -m "refactor: 重构数据加载器"
```

### 2. 分支策略
- `main/master`: 稳定的主分支
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复分支

### 3. 代码审查
```bash
# 创建Pull Request
gh pr create --title "添加新功能" --body "详细描述"

# 审查Pull Request
gh pr review --approve
gh pr merge
```

## 常见问题解决

### 1. 文件太大无法上传
```bash
# GitHub单文件限制100MB，仓库建议小于1GB
# 使用Git LFS管理大文件
git lfs install
git lfs track "*.pth"
git lfs track "*.npy"
git add .gitattributes
```

### 2. 认证问题
```bash
# 如果推送时要求密码，配置Personal Access Token
# 在GitHub Settings > Developer settings > Personal access tokens 生成token
# 使用token作为密码
```

### 3. 撤销提交
```bash
# 撤销最后一次提交（保留文件修改）
git reset --soft HEAD~1

# 撤销最后一次提交（丢弃文件修改）
git reset --hard HEAD~1

# 修改最后一次提交信息
git commit --amend -m "新的提交信息"
```

## 团队协作

### 1. 贡献者指南
创建 `CONTRIBUTING.md` 文件，说明：
- 如何报告bug
- 如何提交功能请求
- 代码风格指南
- 测试要求

### 2. 问题追踪
使用GitHub Issues功能：
- 创建Issue模板
- 使用标签分类问题
- 分配责任人

### 3. 项目管理
使用GitHub Projects：
- 创建看板
- 跟踪进度
- 管理里程碑

## 自动化工作流

### 1. GitHub Actions
创建 `.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
```

## 总结

按照以上步骤，您就可以成功将项目上传到GitHub并进行版本管理。记住：

1. **定期提交**：小步骤、频繁提交
2. **清晰的提交信息**：描述做了什么改变
3. **使用分支**：不要直接在main分支开发
4. **文档先行**：保持README和文档更新
5. **安全考虑**：不要提交敏感信息（密码、API密钥等）

如果遇到问题，可以查看GitHub的官方文档或使用 `git help` 命令获取帮助。