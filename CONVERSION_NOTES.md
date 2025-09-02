# Colab → Local: cambios aplicados

- Se eliminaron dependencias de Colab (`drive.mount`, `google.colab.patches.cv2_imshow`, magics `%...`, `!pip install`).
- Las rutas absolutas (`/content/...`) se reemplazan por rutas relativas basadas en `Path` (`data/raw/...`).
- Visualizaciones con OpenCV en Colab se sustituyen por `matplotlib.pyplot` cuando aplica.
- Parámetros (semillas, paths) se externalizan vía argumentos CLI o variables de entorno si prefieres.
