import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  
  const buttonStyle = {
    backgroundColor: 'white',
    color: '#165688',           
    border: 'none',             
    padding: '12px 24px',       
    margin: '0 10px',          
    fontWeight: 'bold',         
    fontSize: '1.1rem',       
    borderRadius: '8px',        
    textDecoration: 'none',     
    display: 'inline-flex',
    alignItems: 'center',
    cursor: 'pointer'
  };

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        
     
        <div className={styles.buttons} style={{display: 'flex', justifyContent: 'center', marginTop: '2rem'}}>
          
         
          <Link
            to="/docs"
            style={buttonStyle}>
            Documentación
          </Link>

          
          <Link
            to="https://github.com/aitanacuadra/TFG"
            style={buttonStyle}>
            Enlace a GitHub
          </Link>

        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Inicio`}
      description="Documentación oficial del TFG de Generación de Metadatos">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}