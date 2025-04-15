# 如何参与开源项目

1. 选择感兴趣的项目（可参考GitHub标签：good first issue、help wanted等）
2. 阅读项目README和贡献指南
3. 提Issue反馈bug或建议
4. Fork仓库，Clone到本地
5. 新建分支，完成修改
6. Commit并Push到自己的仓库
7. 提交Pull Request（PR）
8. 与维护者沟通，完善代码，等待合并

也可参与文档、翻译、测试等非代码贡献。

---

## Git与GitHub基础教程

### 1. Git基础命令
- `git clone <仓库地址>`：克隆远程仓库到本地
- `git status`：查看当前仓库状态
- `git add <文件>`：将更改添加到暂存区
- `git commit -m "提交说明"`：提交更改到本地仓库
- `git push`：推送本地更改到远程仓库
- `git pull`：拉取远程仓库最新更改

### 2. GitHub使用流程
1. 注册GitHub账号并登录
2. 搜索并找到感兴趣的开源项目
3. 点击“Fork”将项目复制到自己账号下
4. 在自己账号下的仓库页面点击“Code”获取克隆地址
5. 在本地使用`git clone`命令克隆仓库

### 3. 提交PR（Pull Request）流程
1. 在本地新建分支并进行开发：
   ```bash
   git checkout -b my-feature
   # 修改代码
   git add .
   git commit -m "feat: 新增功能描述"
   git push origin my-feature
   ```
2. 在GitHub页面上，选择“Compare & pull request”按钮，填写PR说明，提交PR
3. 等待项目维护者Review，按需修改后合并

### 4. 常见问题与建议
- 提交前请阅读项目的`CONTRIBUTING.md`和`CODE_OF_CONDUCT.md`
- 保持提交记录简洁有意义
- 多与维护者沟通，虚心接受建议

更多详细教程可参考：[Pro Git 中文版](https://git-scm.com/book/zh/v2) 及 [GitHub官方文档](https://docs.github.com/zh)

---

## 进阶：分支管理与冲突解决

### 1. 分支管理
- 建议每个功能/修复新建独立分支，命名规范如`feature/xxx`、`fix/xxx`
- 常用命令：
  - `git branch`：查看本地分支
  - `git checkout -b <分支名>`：新建并切换分支
  - `git switch <分支名>`：切换分支（新版推荐）
  - `git branch -d <分支名>`：删除本地分支
  - `git push origin <分支名>`：推送分支到远程

### 2. 解决冲突
- 冲突常见于多人协作、合并（merge/rebase）时
- 解决流程：
  1. 执行`git pull`或`git merge`时出现冲突，Git会标记冲突文件
  2. 手动编辑冲突文件，保留正确内容，删除冲突标记
  3. 使用`git add <文件>`标记已解决
  4. 继续`git commit`或`git merge --continue`
- 可用工具：VS Code自带Git冲突可视化、`git status`辅助查看

### 3. 常见建议
- 合并前先`git pull --rebase`保持分支最新，减少冲突
- 冲突解决后务必测试代码，确保功能正常
- 保持分支简洁，及时删除无用分支

更多进阶内容可参考：[Pro Git](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%AE%A1%E7%90%86) 和 [GitHub冲突解决官方文档](https://docs.github.com/zh/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts)

---

## 图示与实际案例

### 1. GitHub开源贡献流程图

![GitHub开源贡献流程](https://user-images.githubusercontent.com/25154432/115995760-2b2e2e80-a5f2-11eb-8e2e-2e7e7e2e7e2e.png)

> 图片来源：[First Contributions 项目](https://github.com/firstcontributions/first-contributions)

### 2. 实际案例：为文档项目提交PR

假设你想为 [document-style-guide](https://github.com/ruanyf/document-style-guide) 项目修正一个错别字：

1. **Fork** 项目到自己账号：[点击这里Fork](https://github.com/ruanyf/document-style-guide/fork)
2. **Clone** 到本地：
   ```bash
   git clone https://github.com/你的用户名/document-style-guide.git
   cd document-style-guide
   ```
3. **新建分支**：
   ```bash
   git checkout -b fix-typo
   ```
4. **修改文件**，如修正README.md中的错别字
5. **提交更改**：
   ```bash
   git add README.md
   git commit -m "fix: 修正错别字"
   git push origin fix-typo
   ```
6. **在GitHub页面发起PR**：[点击这里新建PR](https://github.com/ruanyf/document-style-guide/compare)
7. 填写说明，等待维护者Review并合并

### 3. 冲突解决图示

![Git冲突解决流程](https://user-images.githubusercontent.com/25154432/115995800-6e889d00-a5f2-11eb-8e2e-2e7e7e2e7e2e.png)

> 图片来源：[Pro Git 中文版](https://git-scm.com/book/zh/v2)

---

## 更多详细教程与资源

- [Pro Git 中文版](https://git-scm.com/book/zh/v2)（分支、冲突、协作等全流程）
- [First Contributions](https://github.com/firstcontributions/first-contributions)（新手友好，含图文教程）
- [GitHub官方文档：创建拉取请求](https://docs.github.com/zh/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- [GitHub官方文档：解决合并冲突](https://docs.github.com/zh/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts)
- [廖雪峰Git教程](https://www.liaoxuefeng.com/wiki/896043488029600)（中文，适合入门）

---

- 返回：[新手友好项目推荐](beginner_projects.md)
- 返回：[开源贡献标准流程](workflow.md)
- 返回：[开源精神与协议](spirit_license.md)
- 返回：[开源与职业发展](career.md)
