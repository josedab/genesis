import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'AutoML Synthesis',
    icon: '🤖',
    description: (
      <>
        One function generates production-ready synthetic data. Genesis analyzes 
        your dataset and automatically selects the optimal method and settings.
      </>
    ),
  },
  {
    title: 'Privacy Guaranteed',
    icon: '🔒',
    description: (
      <>
        Built-in differential privacy, k-anonymity, and privacy attack testing. 
        Verify GDPR, HIPAA, and CCPA compliance with automated audits.
      </>
    ),
  },
  {
    title: 'Multiple Data Types',
    icon: '📊',
    description: (
      <>
        Generate tabular data, time series, text, and multi-table databases. 
        Support for conditional generation and domain-specific patterns.
      </>
    ),
  },
  {
    title: 'Quality Evaluation',
    icon: '✅',
    description: (
      <>
        Comprehensive metrics for statistical fidelity, ML utility, and privacy. 
        Beautiful reports with per-column analysis and recommendations.
      </>
    ),
  },
  {
    title: 'GPU Accelerated',
    icon: '⚡',
    description: (
      <>
        Train on GPU with mixed precision for 10x faster generation. 
        Distributed training scales to massive datasets across clusters.
      </>
    ),
  },
  {
    title: 'Production Pipelines',
    icon: '🚀',
    description: (
      <>
        Build reproducible workflows with Pipeline API. Version datasets, 
        detect drift, and stream synthetic data in real-time.
      </>
    ),
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className="text--center margin-bottom--lg">
          Everything You Need for Synthetic Data
        </Heading>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
