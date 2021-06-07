/* eslint-disable */
import React, { useState } from 'react';
import CanvasFreeDrawing from "canvas-free-drawing";
import * as tf from '@tensorflow/tfjs';
import { MnistData } from './data';
import { PROPS } from './const';

const initState = { model: false, canvas: false };
let model: tf.LayersModel;
let cfd: CanvasFreeDrawing;
function App() {
  const [answer, setAnswer] = useState(0);
  const [showAlph, setShowAlph] = useState(true);
  const sendImg = () => {
    // console.log(state.cfd.save());
    const img = new Image();
    img.onload = (el) => {
      const datasetBytesBuffer = new ArrayBuffer(PROPS.Size * 4);
      const canvas = document.createElement('canvas');
      canvas.width = PROPS.W;
      canvas.height = PROPS.H;
      const ctx = canvas.getContext('2d');
      // img.width and img.height will contain the original dimensions
      if (ctx && el) {
        //draw in canvas
        ctx.drawImage(img, 0, 0, PROPS.W, PROPS.H);
        const datasetBytesView = new Float32Array(datasetBytesBuffer, 0, PROPS.Size);
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
            [1, PROPS.W, PROPS.H, 1]);
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
      {showAlph && <div style={{ overflowX: "scroll", fontSize: "3em", width: "100%" }}>{PROPS.classes.map(c => c.kat)}</div>}
      <button onClick={() => setShowAlph(!showAlph)}>{showAlph ? "Hide letters" : "Show all letters"}</button>
      <div >
        <div style={{ display: "inline-block", margin: "2em" }}>
          Draw here: <br />
          <canvas id="cfd" style={{ border: "solid 1px black" }}></canvas>
        </div>
        <div style={{ display: "inline-block", margin: "2em", }}>
          Input to the AI ({PROPS.W}x{PROPS.H})px: <br />
          <img id="img2" style={{ border: "solid 1px black", width: 350, height: 350 }}></img>
        </div>
        <div style={{ display: "inline-block", margin: "2em" }}>
          Example of what the AI was trained on: <br />
          <img id="img3" src="./out.png" style={{ border: "solid 1px black", display: "inline-block", width: 350, height: 350 }}></img>
        </div>
      </div>
      <div >
        <button onClick={clear}>Clear</button>
        <button onClick={undo}>Undo</button>
      </div>
      <br />
      <button onClick={sendImg}>{loading ? "Loading" : "Get answer"}</button>
      <h2>Answer: {PROPS.classes[answer].kat}</h2>
    </div>
  );
}

export default App;
