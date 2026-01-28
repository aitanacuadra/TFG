import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          {/* Botón 1: Documentación (Este se queda igual) */}
          <Link
            className="button button--secondary button--lg"
            to="/docs">
            📚 Ver Documentación
          </Link>

          {/* Botón 2: NUEVO BOTÓN UPM */}
          <Link
            className="button button--lg"
            // 👇 Estilos inline para que el botón sea blanco y el logo se ajuste bien
            style={{
              marginLeft: '15px',
              backgroundColor: 'white',
              color: '#165688', // Color azul UPM para el texto
              border: '1px solid rgba(255,255,255,0.8)',
              display: 'inline-flex',
              alignItems: 'center',
              paddingTop: '0.8rem',
              paddingBottom: '0.8rem'
            }}
            to="https://github.com/aitanacuadra/TFG"> 
            
           
            
            
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