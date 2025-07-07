# 部署指南

## 📋 部署到 GitHub Pages 的完整步骤

### 1. 准备工作

#### 前置条件
- Git 已安装
- GitHub 账户
- 项目已推送到 GitHub

#### 检查项目结构
确保你的项目包含以下文件：
```
CS-Survive-henu.github.io/
├── _config.yml
├── Gemfile
├── index.md
├── _wiki/
├── .github/workflows/deploy.yml
└── ...
```

### 2. GitHub Pages 设置

#### 步骤 1: 访问仓库设置
1. 打开 GitHub 仓库页面
2. 点击 "Settings" 选项卡
3. 滚动到 "Pages" 部分

#### 步骤 2: 配置 Pages
1. **Source**: 选择 "GitHub Actions"
2. **Branch**: 确保是 `main` 或 `master` 分支
3. 点击 "Save"

### 3. 自动部署

#### GitHub Actions 自动部署
我们已经配置了 GitHub Actions，每次推送代码都会自动部署：

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    # ... 详细配置见 deploy.yml
```

#### 手动触发部署
```bash
# 推送代码触发自动部署
git add .
git commit -m "Update content"
git push origin main
```

### 4. 验证部署

#### 检查部署状态
1. 在 GitHub 仓库中点击 "Actions" 选项卡
2. 查看最新的 workflow 运行状态
3. 确保显示绿色的 ✅ 成功标志

#### 访问网站
- 网站地址：`https://cs-survive-henu.github.io`
- 部署完成后 1-2 分钟即可访问

### 5. 常见问题解决

#### 问题 1: 部署失败
**症状**: GitHub Actions 显示红色 ❌
**解决方案**:
1. 检查 `_config.yml` 语法是否正确
2. 确保所有 Markdown 文件格式正确
3. 查看 Actions 日志详细错误信息

#### 问题 2: 网站访问 404
**症状**: 打开网站显示 404 错误
**解决方案**:
1. 确保 `index.md` 文件存在
2. 检查 Pages 设置中的 Source 配置
3. 等待 5-10 分钟让 DNS 生效

#### 问题 3: 样式丢失
**症状**: 网站内容显示但样式不正确
**解决方案**:
1. 检查 `_config.yml` 中的 `baseurl` 设置
2. 确保 CSS 文件路径正确
3. 检查 `assets/css/custom.css` 是否存在

#### 问题 4: 内容更新不生效
**症状**: 推送了新内容但网站没有更新
**解决方案**:
1. 检查 GitHub Actions 是否成功运行
2. 清除浏览器缓存
3. 等待 CDN 缓存更新（最多 24 小时）

### 6. 本地开发测试

#### 安装依赖
```bash
# 使用部署脚本
./deploy.sh install

# 或手动安装
gem install bundler
bundle install
```

#### 本地运行
```bash
# 使用部署脚本
./deploy.sh dev

# 或手动运行
bundle exec jekyll serve --watch --incremental
```

#### 本地访问
打开浏览器访问 `http://localhost:4000`

### 7. 自定义域名（可选）

#### 步骤 1: 购买域名
- 推荐平台：阿里云、腾讯云、GoDaddy 等

#### 步骤 2: 配置 DNS
在域名服务商处添加以下记录：
```
类型: CNAME
名称: www (或 @)
值: cs-survive-henu.github.io
```

#### 步骤 3: 配置 GitHub Pages
1. 在仓库根目录创建 `CNAME` 文件
2. 文件内容为你的域名，如：`www.yourdomain.com`
3. 在 GitHub Pages 设置中启用 "Enforce HTTPS"

### 8. 性能优化

#### 图片优化
- 使用 WebP 格式
- 压缩图片大小
- 使用 CDN 加速

#### CSS/JS 优化
- 压缩 CSS 文件
- 合并 JS 文件
- 启用 Gzip 压缩

#### SEO 优化
- 添加 meta 标签
- 配置 sitemap.xml
- 优化页面标题和描述

### 9. 监控和分析

#### GitHub Pages 使用分析
- 查看 GitHub Insights
- 监控访问量和来源

#### 错误监控
- 配置 404 页面
- 监控 broken links
- 使用 Google Search Console

### 10. 维护更新

#### 定期更新
- 更新 Jekyll 版本
- 更新 Ruby gems
- 检查安全漏洞

#### 内容管理
- 定期更新过时内容
- 添加新的学习资源
- 收集用户反馈

## 🎯 快速命令参考

```bash
# 一键部署命令
./deploy.sh install   # 安装依赖
./deploy.sh dev      # 本地开发
./deploy.sh build    # 构建网站
./deploy.sh deploy   # 部署到 GitHub
./deploy.sh clean    # 清理缓存
```

## 📞 获取帮助

如果遇到任何问题：
1. 查看 [GitHub Issues](https://github.com/CS-Survive-henu/CS-Survive-henu.github.io/issues)
2. 创建新的 Issue 报告问题
3. 联系项目维护者

---

*祝你部署成功！🚀*
