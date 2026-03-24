# HPC 集群项目上传 GitHub 指南

> 适用于无外网的计算节点 + 有外网的登录节点环境

## 前提条件

- 登录节点（admin）可以访问外网
- 已有 SSH 密钥对（`~/.ssh/id_rsa` + `~/.ssh/id_rsa.pub`）

---

## 一、生成 SSH 密钥（如果没有）

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 一路回车即可，密钥默认保存在 ~/.ssh/id_rsa
```

查看公钥：

```bash
cat ~/.ssh/id_rsa.pub
```

## 二、添加 SSH 公钥到 GitHub

1. 复制上一步输出的公钥内容
2. 打开 https://github.com/settings/keys
3. 点击 **New SSH key**
4. Title 随意填写，Key 粘贴公钥，点 **Add SSH key**

## 三、配置 SSH

编辑 `~/.ssh/config`，添加 GitHub 配置：

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
```

验证连接：

```bash
ssh -T git@github.com
# 成功输出: Hi <用户名>! You've successfully authenticated...
```

## 四、准备 .gitignore

在项目根目录创建 `.gitignore`，排除大文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/

# 大文件 / 模型 / 数据
checkpoints/
data/
pretrained_models/
Merged_data/
log/
*.pt
*.pth
*.bin
*.ckpt
*.safetensors
*.h5
*.tar.gz
*.zip

# Slurm
slurm-*.out
*.err

# 编辑器
.DS_Store
.idea/
.vscode/
*.swp
```

> GitHub 单文件限制 100MB，仓库建议不超过 1GB。用 `du -sh */` 检查各目录大小。

## 五、初始化仓库并提交

```bash
cd /path/to/your/project

# 初始化 git 仓库
git init

# 添加所有文件（.gitignore 中的会自动排除）
git add .

# 检查将要提交的文件，确认没有大文件
git status

# 提交
git commit -m "Initial commit"
```

## 六、在 GitHub 创建远程仓库

1. 打开 https://github.com/new
2. 填写仓库名称
3. 选择 Public 或 Private
4. **不要勾选** "Add a README"、".gitignore"、"License"（保持空仓库）
5. 点 **Create repository**

## 七、推送代码

> ⚠️ 必须在**登录节点**（有外网的节点）上执行，不能在计算节点（gpu01 等）上执行

```bash
# 添加远程仓库（使用 SSH 地址，不要用 HTTPS）
git remote add origin git@github.com:<用户名>/<仓库名>.git

# 推送
git push -u origin master
```

如果之前误设了 HTTPS 地址，可以修改：

```bash
git remote set-url origin git@github.com:<用户名>/<仓库名>.git
```

## 八、后续更新

```bash
# 在项目目录下
git add .
git commit -m "描述你的更改"

# 切换到登录节点执行 push
git push
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `Could not resolve host: github.com` | 在无外网的计算节点上执行 | 切换到登录节点执行 push |
| `Permission denied (publickey)` | SSH 公钥未添加到 GitHub | 参考第二步添加公钥 |
| `error: unknown option 'trailer'` | git 版本太老（如 1.8.x） | 用 `-F` 文件方式提交：`echo "msg" > /tmp/msg.txt && git commit -F /tmp/msg.txt` |
| 文件超过 100MB 被拒绝 | GitHub 单文件限制 | 在 `.gitignore` 中排除，或使用 Git LFS |
