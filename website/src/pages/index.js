import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title" style={{fontSize: '3rem'}}>
          {siteConfig.title}
        </h1>
        <p className="hero__subtitle" style={{fontSize: '1.5rem', marginBottom: '20px'}}>
          {siteConfig.tagline}
        </p>
        
        {/* Bloque de datos académicos */}
        <div style={{
            background: 'rgba(0,0,0,0.2)', 
            padding: '20px', 
            borderRadius: '10px', 
            marginBottom: '30px',
            maxWidth: '800px',
            margin: '0 auto 30px auto'
        }}>
          <p style={{margin: 0, fontWeight: 'bold'}}>Trabajo de Fin de Grado</p>
          <p style={{margin: '5px 0'}}>Ingeniería de Tecnologías y Servicios de Telecomunicación</p>
          <p style={{margin: 0, fontStyle: 'italic'}}>Universidad Politécnica de Madrid (UPM)</p>
          <hr style={{opacity: 0.3, margin: '15px 0'}} />
          <p style={{margin: 0}}>Autora: <strong>Aitana Cuadra</strong></p>
        </div>

        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs">
            Documentación
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{marginLeft: '10px', color: 'white', borderColor: 'white'}}
            to="https://github.com/aitanacuadra/TFG">
            Ver en GitHub
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
      title="Inicio"
      description="Documentación TFG Aitana Cuadra - UPM">
      <HomepageHeader />
      <main>
        <div className="container" style={{padding: '4rem 2rem', textAlign: 'center'}}>
            <div className="row">
                
            </div>
        </div>
      </main>
    </Layout>
  );
}