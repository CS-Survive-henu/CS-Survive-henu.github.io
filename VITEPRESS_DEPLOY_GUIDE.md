# VitePress 部署指南

本指南详细介绍如何将项目从Jekyll迁移到VitePress并部署到GitHub Pages。

## 📋 迁移概览

### 技术栈变化
- **之前**: Jekyll + Ruby + Liquid模板
- **现在**: VitePress + Vue.js + Markdown

### 主要优势
- ⚡ **更快的构建速度**：VitePress基于Vite，构建速度更快
- 🎨 **更好的主题系统**：现代化的默认主题
- 🔍 **内置搜索**：本地搜索功能
- 📱 **更好的移动端体验**：响应式设计优化
- 🛠️ **更好的开发体验**：热重载、TypeScript支持

## 🚀 部署到GitHub Pages

### 方法一：GitHub Actions自动部署（推荐）

1. **配置GitHub Actions**
   项目已包含 `.github/workflows/vitepress-deploy.yml` 文件

2. **设置GitHub Pages**
   - 进入项目的GitHub仓库
   - 点击 Settings > Pages
   - 在 Source 中选择 "GitHub Actions"

3. **推送代码**
   ```bash
   git add .
   git commit -m "feat: 迁移到VitePress"
   git push origin main
   ```

4. **查看部署状态**
   - 在GitHub仓库的Actions标签页查看部署状态
   - 成功后可在 https://cs-survive-henu.github.io 访问

### 方法二：手动部署

1. **构建项目**
   ```bash
   npm run build
   ```

2. **部署到gh-pages分支**
   ```bash
   npm run deploy
   ```

## 🔧 本地开发环境设置

### 前提条件
确保你的系统安装了：
- Node.js 18.0 或更高版本
- npm 或 yarn

### 环境检查
```bash
# 检查Node.js版本
node --version

# 检查npm版本
npm --version
```

### 安装Node.js（Windows）
如果没有安装Node.js：

1. 访问 [Node.js官网](https://nodejs.org/)
2. 下载LTS版本
3. 运行安装程序
4. 重启命令行工具

### 项目设置
```bash
# 1. 克隆项目
git clone https://github.com/CS-Survive-henu/CS-Survive-henu.github.io.git
cd CS-Survive-henu.github.io

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev

# 4. 在浏览器中访问
# http://localhost:5173
```

## 📁 项目结构

```
CS-Survive-henu.github.io/
├── docs/                          # 文档目录
│   ├── .vitepress/                # VitePress配置
│   │   ├── config.js             # 主配置文件
│   │   └── dist/                 # 构建输出目录
│   ├── guide/                    # 新生指南
│   │   ├── index.md
│   │   ├── toolkit.md
│   │   └── how-to-ask.md
│   ├── ai/                       # AI学习指南
│   ├── competitions/             # 竞赛指北
│   ├── linux/                    # Linux指南
│   ├── opensource/               # 开源贡献
│   ├── career/                   # 升学就业
│   └── index.md                  # 首页
├── .github/                      # GitHub配置
│   └── workflows/
│       └── vitepress-deploy.yml  # 部署工作流
├── package.json                  # 项目依赖
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```

## ⚙️ 配置说明

### VitePress配置
主要配置文件位于 `docs/.vitepress/config.js`：

```javascript
export default defineConfig({
  title: "河南大学计算机生存指北",
  description: "CS-Survive-Henu指南",
  base: '/',
  
  themeConfig: {
    nav: [...],      // 导航栏
    sidebar: {...},  // 侧边栏
    search: {        // 搜索配置
      provider: 'local'
    }
  }
})
```

### 自定义主题
如需自定义主题，可以：
1. 在 `docs/.vitepress/theme/` 目录下创建自定义主题
2. 修改CSS变量来调整样式
3. 添加自定义组件

## 🔍 SEO优化

VitePress自动处理了许多SEO相关的事情：

1. **meta标签**：自动生成基本的meta标签
2. **sitemap**：可配置自动生成sitemap
3. **社交媒体卡片**：支持Open Graph和Twitter Card
4. **结构化数据**：自动添加JSON-LD

### 添加自定义head标签
在config.js中：
```javascript
head: [
  ['link', { rel: 'icon', href: '/favicon.ico' }],
  ['meta', { name: 'keywords', content: '河南大学,计算机,生存指北' }],
  ['meta', { name: 'author', content: '河南大学计算机学院' }]
]
```

## 🚨 常见问题

### Q: 构建时出现内存不足错误
A: 尝试增加Node.js内存限制：
```bash
NODE_OPTIONS="--max-old-space-size=4096" npm run build
```

### Q: 部署后页面空白
A: 检查base路径配置，确保config.js中的base设置正确

### Q: 搜索功能不工作
A: 确保配置了本地搜索：
```javascript
search: {
  provider: 'local'
}
```

### Q: 图片无法显示
A: 确保图片路径正确，建议使用相对路径

## 📊 性能优化

### 构建优化
1. **代码分割**：VitePress自动进行代码分割
2. **静态资源优化**：自动压缩图片和CSS
3. **预渲染**：所有页面都会被预渲染为静态HTML

### 部署优化
1. **CDN加速**：GitHub Pages自带CDN
2. **缓存策略**：合理设置缓存头
3. **压缩传输**：启用gzip压缩

## 🔄 持续集成

### GitHub Actions工作流
自动部署流程：
1. 代码推送到main分支
2. 触发GitHub Actions
3. 安装依赖并构建
4. 部署到GitHub Pages

### 本地预览
在推送前本地预览：
```bash
npm run build
npm run preview
```

## 📝 内容管理

### 添加新页面
1. 在相应目录下创建Markdown文件
2. 在config.js中添加导航和侧边栏配置
3. 提交并推送代码

### 更新内容
1. 直接编辑Markdown文件
2. 使用标准的Markdown语法
3. 支持Vue组件和VitePress扩展语法

## 🎯 下一步计划

1. **多语言支持**：添加英文版本
2. **评论系统**：集成评论功能
3. **数据统计**：添加页面访问统计
4. **内容搜索**：优化搜索功能
5. **移动端优化**：进一步优化移动端体验

## 📞 获取帮助

如果遇到问题，可以：
1. 查看VitePress官方文档
2. 在GitHub上提交Issue
3. 联系项目维护者

---

恭喜！你已经成功将项目迁移到VitePress。享受更好的开发体验吧！🎉
