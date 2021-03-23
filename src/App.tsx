/* eslint-disable */
import React, { useState } from 'react';
import CanvasFreeDrawing from "canvas-free-drawing";
import * as tf from '@tensorflow/tfjs';
import { MnistData } from './data';

const IMAGE_SIZE = 50 * 50;
const initState = { model: false, canvas: false };
let model: tf.LayersModel;
let cfd: CanvasFreeDrawing;
function App() {
  const [answer, setAnswer] = useState(0);
  const sendImg = () => {
    // console.log(state.cfd.save());
    const img = new Image();
    img.onload = (el) => {
      const datasetBytesBuffer = new ArrayBuffer(IMAGE_SIZE * 4);
      const canvas = document.createElement('canvas');
      canvas.width = 50;
      canvas.height = 50;
      const ctx = canvas.getContext('2d');
      // img.width and img.height will contain the original dimensions
      if (ctx && el) {
        //draw in canvas
        ctx.drawImage(img, 0, 0, 50, 50);
        const datasetBytesView = new Float32Array(datasetBytesBuffer, 0, IMAGE_SIZE);
        const img2: any = document.getElementById('img2')
        if (img2) img2.src = canvas.toDataURL();
        // console.log(canvas.toDataURL());
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let j = 0; j < imageData.data.length / 4; j++) {
          // All channels hold an equal value since the image is grayscale, so
          // just read the red channel.
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
        tf.tidy(() => {
          const testImage = new Float32Array(datasetBytesBuffer);
          const xs = tf.tensor4d(
            testImage,
            [1, 50, 50, 1]);
          const output: any = model.predict(xs);
          // const axis = 1;
          const predictions = output.argMax(1).dataSync();
          console.log(predictions);
          setAnswer(predictions);
        })
      }
    }
    img.src = cfd.save();
    // console.log(img);

  }

  const testData = () => {
    const data = new MnistData();
    data.load().then(async () => {
      const testData = await data.getTestData();
      const testResult: any = model.predict(testData.xs);
      const predictions = Array.from(testResult.argMax(1).dataSync());
      const labels = Array.from(testData.labels.argMax(1).dataSync());
      console.log(`Test result: ${predictions}`);
      console.log(`Labels: ${labels}`);
      let correct = 0;
      for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i];
        if (prediction === labels[i]) correct++;
      }
      console.log(`Correct ${correct}/${predictions.length}`);
      console.log(`Correct %  ${correct / predictions.length}`);

    })
  }

  const undo = () => {
    cfd.undo();
  }

  const clear = () => {
    cfd.clear();
  }
  const [state, setState] = React.useState(initState);
  const [loading, setLoading] = React.useState(false);

  if (state === initState) {
    // console.log(document.URL + 'model/model.json');

    tf.loadLayersModel(document.URL + 'model/model.json')
      .then(fileModel => {
        model = fileModel;
        // model.compile({});
        setState({ ...state, model: true });
        console.log(model);
      });
    setTimeout(() => {
      cfd = new CanvasFreeDrawing({
        elementId: 'cfd',
        width: 350,
        height: 350,
        lineWidth: 12
      });

      // set properties
      cfd.setBackground([0, 0, 0]),
        cfd.setStrokeColor([255, 255, 255]); // in RGB

      setState({ ...state, canvas: true });
    }, 100);
  }

  return (
    <div >
      <header >
        <div >
          <button onClick={clear}>Clear</button>
          <button onClick={undo}>Undo</button>
        </div>
        <h2>Answer: {answer}</h2>
        <div >
          <canvas id="cfd" style={{ border: "solid 1px black", display: "inline-block", margin: "2em" }}></canvas>
          <img id="img2" style={{ border: "solid 1px black", display: "inline-block", margin: "2em", width: 350, height: 350 }}></img>
          <img id="img3" src="./out.png" style={{ border: "solid 1px black", display: "inline-block", margin: "2em", width: 350, height: 350 }}></img>
        </div>
        <button onClick={sendImg}>{loading ? "Loading" : "Get answer"}</button>
        <button onClick={testData}>Test Data</button>
      </header>
    </div>
  );
}

export default App;
