# 项目运行指南

## 📋 快速开始

### 1. 环境准备
确保你已安装：
- Node.js 18+ 
- npm 或 yarn

### 2. 检查环境
```bash
# 检查Node.js版本
node --version

# 检查npm版本
npm --version
```

### 3. 安装依赖
```bash
# 进入项目目录
cd CS-Survive-henu.github.io

# 安装依赖
npm install
```

### 4. 启动开发服务器
```bash
# 启动本地开发服务器
npm run dev
```

### 5. 访问网站
在浏览器中打开：`http://localhost:5173`

## 🚀 构建和部署

### 本地构建
```bash
# 构建项目
npm run build

# 预览构建结果
npm run preview
```

### 部署到GitHub Pages
推送到main分支会自动触发GitHub Actions部署：
```bash
git add .
git commit -m "更新内容"
git push origin main
```

## 📁 项目结构说明

```
docs/
├── .vitepress/
│   ├── config.js          # VitePress配置文件
│   └── dist/              # 构建输出目录
├── index.md               # 首页
├── guide/                 # 新生指南
│   ├── index.md
│   ├── toolkit.md
│   └── how-to-ask.md
├── ai/                    # AI学习指南
├── competitions/          # 竞赛指北
├── linux/                 # Linux指南
├── opensource/            # 开源贡献
└── career/                # 升学就业
```

## 🎯 下一步

1. 查看 [VitePress部署指南](./VITEPRESS_DEPLOY_GUIDE.md)
2. 阅读 [项目README](./README.md)
3. 开始编辑内容文件
4. 提交你的贡献

## 💡 需要帮助？

如果遇到问题：
1. 查看VitePress官方文档
2. 检查Node.js和npm版本
3. 删除node_modules重新安装
4. 联系项目维护者
