import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '河南大学计算机生存指北',
  description: '由河南大学计算机学院和软件学院的学生/毕业生共同维护的开源生存指南',
  lang: 'zh-CN',
  base: '/',
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }],
    ['meta', { name: 'description', content: '河南大学计算机生存指北 - 开源生存指南' }],
    ['meta', { name: 'keywords', content: '河南大学,计算机,生存指北,学习指南,编程,AI,Linux,开源' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css' }]
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'CS-Survive-Henu',
    nav: [
      { text: '首页', link: '/' },
      { text: '序', link: '/序' },
      { text: '新生工具箱', link: '/新生工具箱' },
      { text: 'AI方向', link: '/AI方向' },
      { text: '竞赛指北', link: '/竞赛指北' },
      { text: 'Linux基础', link: '/Linux基础' },
      { text: '升学指导', link: '/升学（保研，考研，or出国）' },
      { text: '实验室', link: '/实验室' },
      { text: '更多', items: [
        { text: '学会提问', link: '/学会提问' },
        { text: '如何使用AI', link: '/如何使用AI' },
        { text: '成为有梦想的CS人', link: '/试着去成为一个有梦想的CS人' },
        { text: 'HCSG', link: '/HCSG‐大学开始的地方' },
        { text: 'Wiki史', link: '/Wiki史' }
      ]}
    ],
    sidebar: {
      '/': [
        {
          text: '📖 序言',
          items: [
            { text: '序', link: '/序' }
          ]
        },
        {
          text: '🎯 快速开始',
          items: [
            { text: '新生工具箱', link: '/新生工具箱' },
            { text: 'AI方向', link: '/AI方向' },
            { text: '竞赛指北', link: '/竞赛指北' },
            { text: 'Linux基础', link: '/Linux基础' }
          ]
        },
        {
          text: '📚 学习指导',
          items: [
            { text: '升学指导', link: '/升学（保研，考研，or出国）' },
            { text: '实验室', link: '/实验室' },
            { text: '学会提问', link: '/学会提问' },
            { text: '如何使用AI', link: '/如何使用AI' }
          ]
        },
        {
          text: '🌟 成长感悟',
          items: [
            { text: '试着去成为一个有梦想的CS人', link: '/试着去成为一个有梦想的CS人' },
            { text: 'HCSG-大学开始的地方', link: '/HCSG‐大学开始的地方' },
            { text: 'Wiki史', link: '/Wiki史' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/CS-Survive-henu/CS-Survive-henu.github.io' }
    ],
    footer: {
      message: '由河南大学计算机学院和软件学院的学生/毕业生共同维护',
      copyright: 'Copyright © 2024 CS-Survive-Henu'
    },
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },
    outline: {
      label: '页面导航',
      level: [2, 3]
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    search: {
      provider: 'local'
    },
    editLink: {
      pattern: 'https://github.com/CS-Survive-henu/CS-Survive-henu.github.io/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    }
  },
  markdown: {
    lineNumbers: true,
    image: {
      lazyLoading: true
    },
    math: true
  }
})
