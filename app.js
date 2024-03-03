// Importa el módulo tensorflow para Node.js
import "@tensorflow/tfjs-node";
// Importa face-api.js que proporciona herramientas para el reconocimiento facial
import * as faceapi from "@vladmandic/face-api";
// Importa el módulo fs para interactuar con el sistema de archivos
import * as fs from "fs";
// Importa el módulo path para manejar rutas de archivos
import * as path from "path";
// Importa el módulo canvas, que permite manipular imágenes en el servidor
import * as canvas from "canvas";

// Extrae las clases Canvas, Image, y ImageData del módulo canvas
const { Canvas, Image, ImageData } = canvas;
// Sobrescribe el entorno para que las funciones de face-api.js funcionen con Canvas, Image y ImageData
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Define una función asincrónica llamada 'run' para realizar el proceso de reconocimiento facial
async function run() {
  // Carga los modelos necesarios para el reconocimiento facial desde el disco
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");

  // Directorio donde se almacenan las imágenes de referencia
  const savedDir = "./db_fotos";
  // Nombre del archivo de imagen a consultar
  const queryImg = "consulta.jpg";
  // Carga la imagen de consulta en un objeto Canvas
  const queryCanvas = await canvas.loadImage(queryImg);

  // Detecta la cara en la imagen de consulta y extrae los puntos característicos y descripciones faciales
  const resultsQuery = await faceapi
    .detectSingleFace(queryCanvas) // Detecta una sola cara en la imagen
    .withFaceLandmarks() // Extrae los puntos característicos de la cara
    .withFaceDescriptor(); // Extrae la descripción facial de la cara

  // Crea un objeto FaceMatcher para comparar las caras detectadas con las caras almacenadas
  const faceMatcher = new faceapi.FaceMatcher(resultsQuery);

  // Imprime un mensaje indicando que se está buscando la persona en las imágenes guardadas...
  console.log("Buscando persona en las imágenes guardadas...");

  // Lee el directorio donde se almacenan las imágenes de referencia
  fs.readdir(savedDir, async (err, files) => {
    if (err) {
      // Imprime un mensaje de error si no se puede leer el directorio
      console.error("Error al leer el directorio:", err);
      return;
    }

    // Itera sobre cada archivo en el directorio de imágenes guardadas
    for (const file of files) {
      // Construye la ruta completa del archivo
      const filePath = path.join(savedDir, file);
      // Carga la imagen guardada en un objeto Canvas
      const savedCanvas = await canvas.loadImage(filePath);
      // Detecta todas las caras en la imagen guardada y extrae sus puntos característicos y descripciones faciales
      const resultSaved = await faceapi
        .detectAllFaces(savedCanvas) // Detecta todas las caras en la imagen
        .withFaceLandmarks() // Extrae los puntos característicos de las caras
        .withFaceDescriptors(); // Extrae las descripciones faciales de las caras

      // Comprueba si se encontraron caras en la imagen guardada y en la imagen de consulta
      if (resultSaved.length > 0 && resultsQuery) {
        // Itera sobre cada resultado de cara detectada en la imagen guardada
        resultSaved.forEach((fd) => {
          // Encuentra la mejor coincidencia entre la cara detectada en la imagen guardada y la cara de la imagen de consulta
          const bestMatch = faceMatcher.findBestMatch(fd.descriptor);
          // Comprueba si la coincidencia tiene una etiqueta diferente a "unknown"
          if (bestMatch.label !== "unknown") {
            // Imprime un mensaje indicando que la persona existe en la imagen guardada
            console.log(`La persona existe en la imagen: ${file}`);
          } else {
            // Imprime un mensaje indicando que la persona no existe en la imagen guardada
            console.log(`La persona NO existe en la imagen: ${file}`);
          }
        });
      }
    }
  });
}

// Llama a la función 'run' para iniciar el proceso de reconocimiento facial
run();
