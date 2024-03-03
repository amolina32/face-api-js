import "@tensorflow/tfjs-node";
import * as faceapi from "@vladmandic/face-api";
import * as fs from "fs";
import * as path from "path";
import * as canvas from "canvas";

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function run() {
  // Cargar modelos pre-entrenados
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");

  const savedDir = "./db_fotos";
  const queryImg = "consulta.jpg";
  const queryCanva = await canvas.loadImage(queryImg);

  // Detectar rostro y landmarks en la imagen de consulta
  const resultsQuery = await faceapi
    .detectSingleFace(queryCanva)
    .withFaceLandmarks()
    .withFaceDescriptor();

  console.log("Buscando persona en las imágenes guardadas...");

  // Leer todas las imágenes guardadas en el directorio
  fs.readdir(savedDir, async (err, files) => {
    if (err) {
      console.error("Error al leer el directorio:", err);
      return;
    }

    // Iterar sobre cada archivo en el directorio
    for (const file of files) {
      const filePath = path.join(savedDir, file);
      const savedCanva = await canvas.loadImage(filePath);

      // Detectar rostro y landmarks en la imagen guardada
      const resultSaved = await faceapi
        .detectSingleFace(savedCanva)
        .withFaceLandmarks()
        .withFaceDescriptor();

      // Verificar si se detectó un rostro en la imagen guardada
      if (resultSaved && resultsQuery) {
        // Crear un FaceMatcher con el rostro de la imagen guardada
        const faceMatcher = new faceapi.FaceMatcher([resultSaved]);

        // Encontrar la mejor coincidencia de rostros en la imagen de consulta
        const bestMatch = faceMatcher.findBestMatch(resultsQuery.descriptor);

        // Imprimir el resultado
        if (bestMatch.label !== "unknown") {
          console.log(`La persona existe en la imagen: ${file}`);
        } else {
          console.log(`La persona NO existe en la imagen: ${file}`);
        }
      }
    }
  });
}

run();
