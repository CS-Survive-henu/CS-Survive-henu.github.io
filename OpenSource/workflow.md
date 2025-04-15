# 开源贡献标准流程

## 完整贡献流程

1. **Fork项目到自己账号**
   - 访问目标项目的GitHub页面
   - 点击右上角的"Fork"按钮
   - 等待Fork完成，页面会自动跳转到你账号下的副本

2. **Clone到本地开发环境**
   ```bash
   # 复制你Fork后的仓库URL
   git clone https://github.com/你的用户名/项目名.git
   cd 项目名
   
   # 添加原始仓库作为upstream，便于同步更新
   git remote add upstream https://github.com/原始作者/项目名.git
   ```

3. **创建并切换到新分支**
   ```bash
   # 确保本地代码是最新的
   git fetch upstream
   git checkout main  # 或master，取决于项目
   git merge upstream/main
   
   # 创建新分支
   git checkout -b feature/your-feature-name
   ```
   > 分支命名建议：使用`feature/xxx`（新功能）、`fix/xxx`（修复Bug）、`docs/xxx`（文档更新）等前缀

4. **进行开发工作**
   - 修改代码、文档或其他内容
   - 确保修改符合项目的代码风格和贡献准则
   - 编写或更新相关测试（如果项目有测试要求）

5. **提交变更**
   ```bash
   # 查看更改
   git status
   git diff
   
   # 添加更改的文件
   git add 文件名  # 添加特定文件
   # 或
   git add .      # 添加所有更改
   
   # 提交更改
   git commit -m "type: 简明扼要的变更描述"
   ```
   > 提交信息规范：许多项目遵循[约定式提交](https://www.conventionalcommits.org/)，常见前缀有`feat:`、`fix:`、`docs:`、`style:`、`refactor:`、`test:`、`chore:`等

6. **推送到远程仓库**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**
   - 访问你Fork的GitHub仓库页面
   - 通常会看到"Compare & pull request"按钮，点击它
   - 填写PR标题和描述，说明你的更改解决了什么问题、如何解决的
   - 点击"Create pull request"提交

8. **参与代码审查**
   - 项目维护者会审查你的PR并可能提出修改建议
   - 根据反馈进行必要的修改：
     ```bash
     # 进行更改后
     git add .
     git commit -m "fix: 根据反馈修改xxx"
     git push origin feature/your-feature-name
     ```
   - 修改会自动更新到已提交的PR中

9. **PR被合并**
   - 维护者满意后会合并你的PR
   - 合并后，你会收到通知
   - 恭喜！你已成功为开源项目做出贡献

10. **清理工作**
    ```bash
    # 切回主分支
    git checkout main
    
    # 同步上游更新
    git fetch upstream
    git merge upstream/main
    
    # 删除已完成的特性分支
    git branch -d feature/your-feature-name
    ```

## 常见问题与解决方案

### 1. 处理合并冲突

如果你的PR与目标分支有冲突，需要解决冲突：

```bash
# 同步上游变更
git fetch upstream
git checkout feature/your-feature-name
git merge upstream/main

# 此时如有冲突，Git会提示，编辑冲突文件
# 在冲突文件中，查找并修复 <<<<<<<, =======, >>>>>>> 标记的部分

# 解决冲突后
git add .
git commit -m "fix: 解决合并冲突"
git push origin feature/your-feature-name
```

### 2. 撤销错误提交

如果需要撤销最近的提交：

```bash
# 撤销最后一次提交，但保留更改
git reset --soft HEAD~1

# 撤销最后一次提交，丢弃更改（谨慎使用）
git reset --hard HEAD~1
```

### 3. 重写提交历史

如果需要整理多个小提交：

```bash
# 合并最近的n个提交
git rebase -i HEAD~n

# 交互式界面中，将除第一个"pick"之外的其他行改为"squash"或"s"
# 保存并关闭编辑器，然后编辑合并后的提交信息
```

> 注意：只对未推送到远程的提交执行重写历史操作，否则需要强制推送(`git push -f`)，这可能会影响其他协作者

## 最佳实践

1. **频繁同步上游**：定期从原始仓库拉取最新更改，减少冲突可能
2. **遵循项目规范**：阅读`CONTRIBUTING.md`或贡献指南，遵循代码风格
3. **小批量提交**：每个PR保持功能聚焦，便于审查和合并
4. **编写清晰的PR描述**：说明目的、实现方式、测试方法等
5. **及时回应反馈**：及时处理审查意见，保持PR活跃
6. **添加测试**：为你的更改编写测试，提高代码质量
7. **使用Issue跟踪**：先创建Issue讨论，再提交PR解决

## 实际案例：修复文档错误

假设我们发现React文档中的一个拼写错误，以下是贡献流程：

1. **Fork React仓库**：在[React GitHub页面](https://github.com/facebook/react)点击Fork

2. **Clone到本地**：
   ```bash
   git clone https://github.com/你的用户名/react.git
   cd react
   git remote add upstream https://github.com/facebook/react.git
   ```

3. **创建分支**：
   ```bash
   git checkout -b docs/fix-typo
   ```

4. **修复错误**：找到并修正文档中的拼写错误

5. **提交更改**：
   ```bash
   git add docs/src/pages/tutorial.md
   git commit -m "docs: fix typo in tutorial page"
   git push origin docs/fix-typo
   ```

6. **创建PR**：在GitHub上创建PR，标题为"Fix typo in tutorial page"，描述中简要说明修复的错误

7. **等待审查并合并**：React团队成员审查后会合并你的PR

## 相关资源

- [如何参与开源项目](how_to_contribute.md)
- [新手友好项目推荐](beginner_projects.md)
- [开源与职业发展](career.md)

---

> 最后更新: 2025年4月15日
