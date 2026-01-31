import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Genesis',
  tagline: 'Privacy-safe synthetic data for ML training, testing, and development',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://genesis-synth.github.io',
  baseUrl: '/genesis/',

  organizationName: 'genesis-synth',
  projectName: 'genesis',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/genesis-synth/genesis/tree/main/website/',
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/genesis-synth/genesis/tree/main/website/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    image: 'img/genesis-social-card.png',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'v1.2.0',
      content: '🎉 Genesis v1.2.0 is out! Plugin system, auto-tuning, privacy certificates, and more. <a href="/docs/changelog">See what is new</a>',
      backgroundColor: '#4f46e5',
      textColor: '#ffffff',
      isCloseable: true,
    },
    navbar: {
      title: 'Genesis',
      logo: {
        alt: 'Genesis Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/reference',
          label: 'API',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          href: 'https://github.com/genesis-synth/genesis',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started'},
            {label: 'Core Concepts', to: '/docs/concepts/overview'},
            {label: 'Examples', to: '/docs/examples'},
          ],
        },
        {
          title: 'API',
          items: [
            {label: 'Reference', to: '/docs/api/reference'},
            {label: 'v1.2.0 Features', to: '/docs/api/v120-features'},
            {label: 'Configuration', to: '/docs/api/configuration'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub Discussions', href: 'https://github.com/genesis-synth/genesis/discussions'},
            {label: 'Stack Overflow', href: 'https://stackoverflow.com/questions/tagged/genesis-synth'},
            {label: 'Contributing', to: '/docs/contributing'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'Blog', to: '/blog'},
            {label: 'Changelog', to: '/docs/changelog'},
            {label: 'GitHub', href: 'https://github.com/genesis-synth/genesis'},
            {label: 'PyPI', href: 'https://pypi.org/project/genesis-synth/'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Genesis. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
