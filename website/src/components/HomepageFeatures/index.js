import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Importamos las imágenes SVG
// El '.default' al final es IMPORTANTE en Docusaurus para que funcione el SVG
const FeatureList = [
  {
    title: 'Inteligencia Artificial',
    
    Svg: require('@site/static/img/ai-agent.svg').default,
    description: (
      <>
        Uso de modelos LLM avanzados para analizar automáticamente el contenido 
        semántico de tus archivos CSV y JSON.
      </>
    ),
  },
  {
    title: 'Estándar DCAT-AP',
    Svg: require('@site/static/img/chat-with-ai.svg').default,
    description: (
      <>
        Generación de metadatos validados y totalmente compatibles con la 
        normativa europea de Datos Abiertos.
      </>
    ),
  },
  {
    title: 'API FastAPI',
    Svg: require('@site/static/img/dev-environment.svg').default,
    description: (
      <>
        Backend de alto rendimiento construido con Python y FastAPI, 
        documentado automáticamente y fácil de desplegar.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {/* Aquí renderizamos el SVG con una clase CSS para controlar el tamaño */}
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}