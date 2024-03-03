import "@tensorflow/tfjs-node";
import * as faceapi from "@vladmandic/face-api";
import * as canvas from "canvas";

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function run() {
  // Cargar modelos pre-entrenados
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");

  const savedImg = "guardada3.jpeg";
  const queryImg = "consulta.jpeg";
  const savedCanva = await canvas.loadImage(savedImg);
  const queryCanva = await canvas.loadImage(queryImg);

  // Detectar rostros y landmarks en la imagen guardada
  const resultSaved = await faceapi
    .detectAllFaces(savedCanva)
    .withFaceLandmarks()
    .withFaceDescriptors();

  // Detectar un solo rostro y landmarks en la imagen de consulta
  const resultsQuery = await faceapi
    .detectSingleFace(queryCanva)
    .withFaceLandmarks()
    .withFaceDescriptor();

  // Verificar si se detectaron rostros en alguna de las imágenes
  if (resultSaved.length === 0 || !resultsQuery) {
    console.log("No se detectaron rostros en una de las imágenes.");
    return;
  }

  // Crear un FaceMatcher con los rostros de la imagen guardada
  const faceMatcher = new faceapi.FaceMatcher(resultSaved);

  // Encontrar la mejor coincidencia de rostros en la imagen de consulta
  const bestMatch = faceMatcher.findBestMatch(resultsQuery.descriptor);

  console.log(bestMatch);

  if (bestMatch.label !== "unknown") {
    console.log(`La persona existe en la imagen guardada: ${bestMatch.label}`);
  } else {
    console.log("La persona no fue encontrada en la imagen guardada.");
  }
}

run();
