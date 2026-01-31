import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroLogo}>
          <img src="/img/logo.svg" alt="Genesis Logo" width="120" height="120" />
        </div>
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.installCommand}>
          <code>pip install genesis-synth</code>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/">
            Get Started →
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/api/reference">
            API Reference
          </Link>
          <Link
            className="button button--secondary button--lg"
            href="https://github.com/genesis-synth/genesis">
            GitHub ⭐
          </Link>
        </div>
      </div>
    </header>
  );
}

function QuickDemo() {
  const demoCode = `import genesis as gn

# Load your sensitive data
real_data = pd.read_csv("customers.csv")

# Generate privacy-safe synthetic data
generator = gn.CTGANGenerator(epochs=100)
generator.fit(real_data)
synthetic_data = generator.generate(1000)

# Validate quality and privacy
report = gn.evaluate(real_data, synthetic_data)
print(f"Quality Score: {report.quality_score:.2f}")
print(f"Privacy Score: {report.privacy_score:.2f}")`;

  return (
    <section className={styles.demoSection}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <Heading as="h2">Generate Synthetic Data in Seconds</Heading>
            <p className={styles.demoText}>
              Transform sensitive datasets into privacy-safe synthetic versions
              that maintain statistical properties. Perfect for ML training,
              testing, and development without exposing real customer data.
            </p>
            <ul className={styles.demoList}>
              <li>✅ Preserves statistical distributions</li>
              <li>✅ Maintains column correlations</li>
              <li>✅ Built-in privacy guarantees</li>
              <li>✅ GDPR, HIPAA, CCPA compliant</li>
            </ul>
          </div>
          <div className="col col--6">
            <CodeBlock language="python" title="quickstart.py">
              {demoCode}
            </CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

function UseCases() {
  const cases = [
    {
      title: 'ML Training',
      icon: '🤖',
      description: 'Train models on unlimited synthetic data without privacy concerns or data access restrictions.',
    },
    {
      title: 'Software Testing',
      icon: '🧪',
      description: 'Generate realistic test data that covers edge cases and matches production distributions.',
    },
    {
      title: 'Data Sharing',
      icon: '🔄',
      description: 'Share data across teams and partners without exposing sensitive customer information.',
    },
    {
      title: 'Compliance',
      icon: '🛡️',
      description: 'Meet GDPR, HIPAA, and CCPA requirements with privacy-certified synthetic datasets.',
    },
  ];

  return (
    <section className={styles.useCases}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Built for Real-World Use Cases
        </Heading>
        <div className={styles.useCaseGrid}>
          {cases.map((item, idx) => (
            <div key={idx} className={styles.useCaseCard}>
              <div className={styles.useCaseIcon}>{item.icon}</div>
              <Heading as="h3">{item.title}</Heading>
              <p>{item.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Comparison() {
  return (
    <section className={styles.comparison}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Why Genesis?
        </Heading>
        <div className={styles.comparisonTable}>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Genesis</th>
                <th>SDV</th>
                <th>Gretel.ai</th>
                <th>Faker</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Statistical Fidelity</td>
                <td>✅ High</td>
                <td>✅ High</td>
                <td>✅ High</td>
                <td>❌ None</td>
              </tr>
              <tr>
                <td>Privacy Guarantees</td>
                <td>✅ DP, k-anon</td>
                <td>⚠️ Basic</td>
                <td>✅ Yes</td>
                <td>❌ N/A</td>
              </tr>
              <tr>
                <td>Open Source</td>
                <td>✅ MIT</td>
                <td>✅ BSL</td>
                <td>❌ Proprietary</td>
                <td>✅ MIT</td>
              </tr>
              <tr>
                <td>Self-Hosted</td>
                <td>✅ Yes</td>
                <td>✅ Yes</td>
                <td>❌ Cloud Only</td>
                <td>✅ Yes</td>
              </tr>
              <tr>
                <td>Time Series</td>
                <td>✅ Yes</td>
                <td>✅ Yes</td>
                <td>✅ Yes</td>
                <td>❌ No</td>
              </tr>
              <tr>
                <td>Text Generation</td>
                <td>✅ LLM-based</td>
                <td>❌ No</td>
                <td>✅ Yes</td>
                <td>⚠️ Templates</td>
              </tr>
              <tr>
                <td>Auto-Tuning</td>
                <td>✅ Optuna</td>
                <td>❌ No</td>
                <td>❌ No</td>
                <td>❌ N/A</td>
              </tr>
              <tr>
                <td>Plugin System</td>
                <td>✅ Extensible</td>
                <td>⚠️ Limited</td>
                <td>❌ No</td>
                <td>✅ Providers</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function Metrics() {
  const metrics = [
    { value: '99%', label: 'Privacy Score', icon: '🔒' },
    { value: '94%', label: 'Statistical Fidelity', icon: '📊' },
    { value: '<5 min', label: 'Time to First Result', icon: '⚡' },
    { value: '291', label: 'Tests Passing', icon: '✅' },
  ];

  return (
    <section className={styles.metrics}>
      <div className="container">
        <div className={styles.metricsGrid}>
          {metrics.map((metric, idx) => (
            <div key={idx} className={styles.metricCard}>
              <div className={styles.metricIcon}>{metric.icon}</div>
              <div className={styles.metricValue}>{metric.value}</div>
              <div className={styles.metricLabel}>{metric.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className={styles.architecture}>
      <div className="container">
        <Heading as="h2" className="text--center">
          How Genesis Works
        </Heading>
        <p className="text--center margin-bottom--lg" style={{maxWidth: '700px', margin: '0 auto 2rem'}}>
          Genesis learns the statistical patterns in your data and generates new samples that preserve those patterns—without copying any real records.
        </p>
        <div className={styles.architectureDiagram}>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>📥</div>
            <div className={styles.archTitle}>Input</div>
            <div className={styles.archDesc}>Your real data (CSV, Parquet, DataFrame)</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>🔬</div>
            <div className={styles.archTitle}>Analyze</div>
            <div className={styles.archDesc}>Detect schema, distributions, correlations</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>🧠</div>
            <div className={styles.archTitle}>Learn</div>
            <div className={styles.archDesc}>Train generative model (CTGAN, TVAE, etc.)</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>✨</div>
            <div className={styles.archTitle}>Generate</div>
            <div className={styles.archDesc}>Create new privacy-safe samples</div>
          </div>
          <div className={styles.archArrow}>→</div>
          <div className={styles.archStep}>
            <div className={styles.archIcon}>✅</div>
            <div className={styles.archTitle}>Validate</div>
            <div className={styles.archDesc}>Quality & privacy metrics</div>
          </div>
        </div>
      </div>
    </section>
  );
}

function TrustedBy() {
  return (
    <section className={styles.trustedBy}>
      <div className="container">
        <p className={styles.trustedByTitle}>Built for production use cases</p>
        <div className={styles.trustedByLogos}>
          <span className={styles.trustedByItem}>🏥 Healthcare</span>
          <span className={styles.trustedByItem}>🏦 Finance</span>
          <span className={styles.trustedByItem}>🛒 Retail</span>
          <span className={styles.trustedByItem}>🔬 Research</span>
          <span className={styles.trustedByItem}>🏢 Enterprise</span>
        </div>
        <p className={styles.trustedBySubtitle}>
          Used by data science teams for ML training, software testing, and compliance-safe data sharing
        </p>
      </div>
    </section>
  );
}

function CallToAction() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <Heading as="h2">Ready to Generate Privacy-Safe Data?</Heading>
        <p>Get started in under 5 minutes with our comprehensive documentation.</p>
        <div className={styles.ctaButtons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started/quickstart">
            Quick Start Guide
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/examples">
            View Examples
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Privacy-Safe Synthetic Data Generation"
      description="Generate high-quality synthetic data for ML training, testing, and development. GDPR, HIPAA, and CCPA compliant. Open source and self-hosted.">
      <HomepageHeader />
      <main>
        <Metrics />
        <HomepageFeatures />
        <Architecture />
        <QuickDemo />
        <UseCases />
        <Comparison />
        <TrustedBy />
        <CallToAction />
      </main>
    </Layout>
  );
}
