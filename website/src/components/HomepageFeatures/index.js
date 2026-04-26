import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';


const FeatureList = [
  {
    title: 'LLMs',
    
    Svg: require('@site/static/img/ai-agent.svg').default,
    description: (
      <>
        Procesamiento automatico de archivos mediante IA para extraer metadatos
      </>
    ),
  },
  {
    title: 'Estándar DCAT-AP',
    Svg: require('@site/static/img/chat-with-ai.svg').default,
    description: (
      <>
        Generación de metadatos validados y totalmente compatibles con la 
        normativa DCAT-AP 
      </>
    ),
  },
  {
    title: 'API FastAPI',
    Svg: require('@site/static/img/dev-environment.svg').default,
    description: (
      <>
        Backend construido con Python y FastAPI
        
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
       
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