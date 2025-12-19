// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'API TFG Metadatos', // 1. Título de tu web
  tagline: 'Generación Automática de Metadatos DCAT/DCAT-AP', // 2. Subtítulo (Tema de tu TFG)
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
  // 3. Configuración para GitHub Pages (¡MUY IMPORTANTE!)
  url: 'https://aitanacuadra.github.io', // Tu usuario + github.io
  baseUrl: '/TFG/', // El nombre de tu repositorio con barras
  
  // GitHub pages deployment config.
  organizationName: 'aitanacuadra', // Tu usuario de GitHub
  projectName: 'TFG', // El nombre de tu repositorio

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // 4. Cambiamos el idioma a español
  i18n: {
    defaultLocale: 'es',
    locales: ['es'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Quitamos el "editUrl" para que no salga el botón de "Editar esta página"
          // que llevaría al repo de Facebook por defecto.
        },
        blog: false, // 5. Desactivamos el blog porque en un TFG no suele hacer falta
        theme: {
          customCss: './src/css/custom.css',
          
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'API TFG', // Título de la barra superior
        logo: {
          alt: 'Logo TFG',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentación', // Renombrado de 'Tutorial' a 'Documentación'
          },
          // He quitado el enlace al Blog
          {
            href: 'https://github.com/aitanacuadra/TFG', // Tu repo real
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentación',
            items: [
              {
                label: 'Introducción',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Contacto',
            items: [
              {
                label: 'GitHub Profile',
                href: 'https://github.com/aitanacuadra',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Aitana Cuadra. Trabajo de Fin de Grado.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;