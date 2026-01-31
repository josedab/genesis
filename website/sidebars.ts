import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/first-generator',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/index',
        'tutorials/customer-data',
        'tutorials/healthcare-compliance',
        'tutorials/ml-pipeline',
        'tutorials/testing',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/overview',
        'concepts/generators',
        'concepts/privacy',
        'concepts/evaluation',
        'concepts/constraints',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        {
          type: 'category',
          label: 'Data Types',
          items: [
            'guides/tabular-data',
            'guides/time-series',
            'guides/text-generation',
            'guides/multi-table',
          ],
        },
        {
          type: 'category',
          label: 'v1.4.0 Features',
          items: [
            'guides/automl',
            'guides/augmentation',
            'guides/privacy-attacks',
            'guides/drift-detection',
            'guides/versioning',
            'guides/pipelines',
            'guides/domain-generators',
          ],
        },
        'guides/conditional-generation',
        'guides/streaming',
        'guides/privacy-compliance',
      ],
    },
    {
      type: 'category',
      label: 'Migration',
      items: [
        'migration/index',
        'migration/from-sdv',
        'migration/from-faker',
        'migration/from-gretel',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/reference',
        'api/configuration',
        'api/cli',
        'api/v130-features',
        'api/v140-features',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'advanced/plugins',
        'advanced/distributed',
        'advanced/gpu',
        'advanced/federated',
      ],
    },
    'examples',
    'benchmarks',
    'troubleshooting',
    'faq',
    'why-genesis',
    'contributing',
    'changelog',
  ],
};

export default sidebars;
