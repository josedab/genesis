import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Genesis',
  tagline: 'Privacy-safe synthetic data generation for ML, testing, and development',
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
          showLastUpdateTime: false,
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
    image: 'img/genesis-social-card.svg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'v1.4.0',
      content: '🚀 Genesis v1.4.0 is here! AutoML synthesis, privacy attack testing, drift detection, and more. <a href="/genesis/docs/changelog">See what\'s new →</a>',
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
          to: '/docs/guides/automl',
          label: 'v1.4.0',
          position: 'left',
          className: 'navbar-v14-badge',
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
            {label: 'Getting Started', to: '/docs/getting-started/installation'},
            {label: 'Core Concepts', to: '/docs/concepts/overview'},
            {label: 'Examples', to: '/docs/examples'},
          ],
        },
        {
          title: 'Guides',
          items: [
            {label: 'AutoML Synthesis', to: '/docs/guides/automl'},
            {label: 'Privacy Testing', to: '/docs/guides/privacy-attacks'},
            {label: 'Domain Generators', to: '/docs/guides/domain-generators'},
          ],
        },
        {
          title: 'API',
          items: [
            {label: 'Reference', to: '/docs/api/reference'},
            {label: 'CLI Commands', to: '/docs/api/cli'},
            {label: 'Configuration', to: '/docs/api/configuration'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/genesis-synth/genesis'},
            {label: 'Discussions', href: 'https://github.com/genesis-synth/genesis/discussions'},
            {label: 'Contributing', to: '/docs/contributing'},
            {label: 'Changelog', to: '/docs/changelog'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Genesis. MIT License. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml', 'toml'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
