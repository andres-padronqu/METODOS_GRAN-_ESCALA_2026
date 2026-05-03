# Producto de Datos de Pronóstico de Ventas

## Proyecto realizado por Andrés Padrón y Manuel De la Tejera

## 1. Problema de negocio

El objetivo de este proyecto es construir un **producto de datos** que permita a las áreas de planeación, finanzas, BI y dirección acceder a pronósticos de ventas de manera sencilla, rápida y sin depender de notebooks o equipos técnicos.

Actualmente, el proceso de generación de pronósticos es manual, lo que limita la capacidad de reacción ante cambios en la demanda. Este proyecto propone una solución basada en la nube que habilita:

- Acceso vía interfaz web
- Consultas rápidas sobre datos históricos y predicciones
- Generación de pronósticos a nivel individual y batch
- Captura de feedback del negocio

---

## 2. Arquitectura de la solución

![Arquitectura](diagrams/arquitectura.png)

La solución se diseñó como un producto de datos desplegado en AWS, separando claramente las capas de aplicación, datos y machine learning.

### Componentes principales

- **Amazon ECS (Fargate)** — Despliegue de la aplicación Streamlit como interfaz web accesible vía URL pública.
- **Amazon ECR** — Repositorio de imágenes Docker para contenerizar la aplicación.
- **Amazon S3** — Data lake con datos históricos, predicciones y métricas de evaluación.
- **AWS Glue Data Catalog** — Catálogo de metadatos que estructura los datos almacenados en S3.
- **Amazon Athena** — Motor de consultas SQL para acceder a los datos en S3.
- **Amazon SageMaker** — Entrenamiento y generación de predicciones de forma offline (batch).
- **Amazon RDS (PostgreSQL)** — Base de datos operacional: feedback, metadatos, logs de uso.
- **AWS Secrets Manager** — Gestión segura de credenciales de la base de datos.
- **AWS CloudFormation** — Despliegue de infraestructura como código (IaC).

### Flujo general

1. Los modelos generan predicciones en SageMaker
2. Las predicciones se almacenan en S3
3. Glue cataloga los datos
4. Athena permite consultas SQL sobre esos datos
5. La aplicación en ECS consume los datos vía Athena
6. Los usuarios interactúan con la app vía navegador
7. El feedback se almacena en RDS

---

## 3. Modelo de datos (RDS)

![ERD](diagrams/ERD.png)

La base de datos relacional en RDS soporta la capa operacional del producto.

### Tablas principales

- **categories** — Catálogo de categorías de productos
- **items** — Información de productos, relación con categorías
- **shops** — Información de tiendas
- **forecasts** — Predicciones generadas, nivel producto-tienda-mes
- **model_metrics** — Métricas de evaluación vs baseline naive
- **business_feedback** — Comentarios del negocio sobre predicciones

### Relaciones

- Un `item` pertenece a una `category`
- Un `item` puede tener múltiples `forecasts`
- Un `shop` puede tener múltiples `forecasts`
- Las métricas se calculan por combinación de item, shop y categoría
- El feedback se vincula a productos, tiendas y categorías

---

## 4. Aplicación

Se desarrolló una aplicación interactiva con **Streamlit**, desplegada en AWS, que permite a usuarios de negocio consumir el producto sin conocimientos técnicos. Las predicciones provienen de un modelo **LightGBM** entrenado con features de lags, ejecutado mediante un pipeline reproducible de preprocessing, training e inference.

---

### Home

![Home](docs/images/app/home.png)

Pantalla de bienvenida del producto. Presenta el propósito del sistema y da acceso a todas las funcionalidades disponibles desde el menú lateral. Es el punto de entrada para cualquier usuario de negocio, sin importar su perfil técnico.

---

### Dashboard ejecutivo

![Dashboard](docs/images/app/dashboard.png)

Vista de alto nivel orientada a perfiles directivos. Muestra la comparación agregada entre ventas reales y pronóstico, el benchmark contra el baseline naive, las métricas globales del modelo (RMSE) y la evolución temporal de las predicciones. Permite al COO y al CFO tener una visión rápida del desempeño del sistema.

---

### Inferencia individual

![Inferencia](docs/images/app/inferencia.png)

Vista de consulta granular. El usuario selecciona una tienda y un producto específico y obtiene el pronóstico del mes siguiente junto con la comparación contra los valores históricos reales. Está orientada a analistas de planeación que necesitan revisar predicciones a nivel operativo.

---

### Generación batch

![Batch](docs/images/app/batch.png)

Permite ejecutar pronósticos masivos seleccionando una categoría completa, una tienda o el catálogo completo. Los resultados se presentan en tabla descargable. Resuelve directamente el caso de uso del Director de Finanzas, que necesita un archivo semanal con pronósticos del mes siguiente para enviarlo al CFO.

---

### KPIs del modelo

![KPIs](docs/images/app/KPIS.png)

Tablero de métricas de evaluación del modelo. Muestra RMSE, MAE, RMSE del baseline naive y la mejora porcentual, con desglose por categoría de producto. Permite al equipo de ML identificar qué segmentos tienen mayor error y requieren atención prioritaria.

---

### Captura de feedback

![Feedback](docs/images/app/feedback.png)

Formulario para que analistas de negocio documenten observaciones sobre predicciones que no parecen correctas. El usuario indica el producto, el tipo de problema (sobreestimación, subestimación, otro) y un comentario en texto libre. El registro se almacena en RDS y queda disponible para el equipo de ML.

---

### Productos problema

![Productos Problema](docs/images/app/productos_problema.png)

Listado consolidado de todos los productos marcados como problemáticos por el negocio. Filtrable por estatus (abierto, en revisión, cerrado). Permite al equipo de ML priorizar qué series de tiempo investigar y cerrar el ciclo de mejora continua del modelo.

---

### Video demo

[Ver demo de la aplicación ]https://drive.google.com/file/d/1Mk1uIBe0FG4dJ2RhAR7bFfPsvy0CF41S/view?usp=sharing

## 5. Despliegue en AWS (ECS + ECR + LightGBM)

La aplicación fue contenerizada con Docker y desplegada en AWS usando ECS con Fargate. El modelo central es **LightGBM**, entrenado para predecir ventas mensuales a nivel producto–tienda.

---

### Evaluación del modelo

| Métrica | Valor |
|---------|-------|
| RMSE modelo | 0.78 |
| RMSE naive | 1.08 |
| **Mejora** | **27.7%** |

---

### Pipeline de modelado

1. **Preprocessing** — limpieza de datos, generación de features y variables lag
2. **Training** — LightGBM con validación temporal, evaluación con RMSE y MAE
3. **Inference** — predicciones generadas y almacenadas en S3

---

### Integración en la arquitectura

**Predicción → Consumo → Feedback → Iteración**

- Predicciones en **S3**, consultadas vía **Athena**
- Visualización en **Streamlit (ECS)**
- Feedback almacenado en **RDS**

---

### CloudFormation — infraestructura como código

Toda la infraestructura del POC fue desplegada mediante una plantilla CloudFormation en un solo stack. La consola muestra todos los recursos en estado `CREATE_COMPLETE`, lo que confirma que el despliegue fue exitoso y reproducible sin intervención manual.

![CloudFormation](docs/images/app/CloudFormation_CreateComplete.png)

---

### Cluster en ECS

Se creó el cluster `streamlit-cluster` en Amazon ECS. Este es el entorno de cómputo que agrupa y administra el servicio de la aplicación Streamlit en Fargate, sin necesidad de gestionar servidores subyacentes.

![ECS Cluster](docs/images/app/ECS_cluster.png)

---

### Servicio y tareas activas

El servicio `streamlit-service` mantiene una tarea Fargate activa en todo momento. La consola confirma que la tarea está en estado `RUNNING`, lo que garantiza que la aplicación está disponible para los usuarios finales.

![ECS Tasks](docs/images/app/ECS_tasks.png)

---

### Configuración de red

El contenedor corre con una IP pública asignada automáticamente por Fargate, expuesta en el puerto `8501`. Esta configuración permite el acceso directo desde cualquier navegador sin necesidad de un load balancer adicional, lo cual es apropiado para el alcance del MVP.

![ECS Networking](docs/images/app/ECS_networking.png)

---

### Logs del contenedor

Los logs del contenedor se envían automáticamente a CloudWatch Logs. Esto permite al equipo de plataforma monitorear el comportamiento de la aplicación en tiempo real, diagnosticar errores y auditar el uso del sistema sin necesidad de acceso SSH al contenedor.

![ECS Logs](docs/images/app/ECS_logs.png)

---

### Imagen en ECR

La imagen Docker de la aplicación fue construida localmente y publicada en Amazon ECR. ECR actúa como repositorio privado de imágenes, asegurando que solo el servicio ECS autorizado por IAM puede hacer pull de la imagen. La imagen está versionada con tags para facilitar rollbacks.

![ECR](docs/images/app/ECR_image.png)

---

### Aplicación desplegada en producción

La aplicación de Streamlit es accesible vía URL pública desde cualquier navegador, sin necesidad de instalar software ni tener acceso a AWS.

**URL pública:** http://54.221.9.247:8501/

---

#### Home

La pantalla de inicio confirma que la app está corriendo en producción sobre ECS Fargate, accesible desde la IP pública del contenedor.

![App Home](docs/images/app/ECS_home.png)

---

#### Dashboard ejecutivo

Vista del dashboard en producción con datos reales del modelo LightGBM. Muestra la comparación entre ventas históricas y pronósticos generados, permitiendo al equipo directivo evaluar el desempeño del sistema de forma inmediata.

![Dashboard](docs/images/app/ECS_dashboard_light.png)

---

#### Inferencia individual

Vista de inferencia en producción. El usuario selecciona una tienda y un producto y obtiene la predicción del modelo LightGBM para el mes siguiente, junto con el histórico de ventas reales para contexto.

![Inferencia](docs/images/app/ECS_inf_lightgbm.png)

---

#### Generación batch

Vista de generación batch en producción. Permite seleccionar una categoría o tienda completa y descarga el archivo con todos los pronósticos del grupo. Los datos provienen de las predicciones pre-computadas almacenadas en S3 y consultadas vía Athena.

![Batch](docs/images/app/ECS_batch_lightgbm.png)

---

#### KPIs del modelo

Vista de KPIs en producción con métricas reales del modelo LightGBM. Confirma un RMSE de 0.78 frente a un naive de 1.08, representando una mejora del 27.7%. El desglose por categoría permite identificar segmentos con mayor error de predicción.

![KPIs](docs/images/app/ECS_KPIS_lightgbm.png)

---

#### Captura de feedback

Vista del formulario de feedback en producción. Los analistas de negocio pueden registrar observaciones directamente desde la interfaz. El registro se almacena en RDS PostgreSQL y queda disponible para el equipo de ML.

![Feedback](docs/images/app/ECS_feedback_lightgbm.png)

---

#### Productos marcados para revisión

Vista del listado de productos problema en producción. Consolida todos los registros de feedback capturados por el negocio, filtrable por estatus. Cierra el ciclo entre el negocio y el equipo de ML.

![Productos Problema](docs/images/app/ECS_prod_lightgbm.png)

---

Este enfoque convierte al modelo en un componente productivo dentro de un sistema de negocio, permitiendo su uso continuo, monitoreo y mejora basada en retroalimentación real.