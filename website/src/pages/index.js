import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  // 1. Estilo común para ambos botones (Blanco y Azul UPM)
  const buttonStyle = {
    backgroundColor: 'white',
    color: '#165688',           // Texto Azul
    border: 'none',             // Sin bordes extraños
    padding: '12px 24px',       // Tamaño del botón
    margin: '0 10px',           // Separación entre ellos
    fontWeight: 'bold',         // Texto en negrita
    fontSize: '1.1rem',         // Tamaño de letra
    borderRadius: '8px',        // Bordes un poco redondeados
    textDecoration: 'none',     // Quitar subrayado
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
        
        {/* Contenedor de botones centrado */}
        <div className={styles.buttons} style={{display: 'flex', justifyContent: 'center', marginTop: '2rem'}}>
          
          {/* Botón 1: Documentación */}
          <Link
            to="/docs"
            style={buttonStyle}>
            Documentación
          </Link>

          {/* Botón 2: GitHub (Idéntico al anterior) */}
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