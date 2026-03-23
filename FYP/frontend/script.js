let topBoxes = [];
let wideBoxes = [];

let selectedTop = null;
let selectedWide = null;

let manualMode = false;
let manualTarget = null;
let startX = null;
let startY = null;

let lastQuery = "";

// handles switching betweeen UI screens by toggling 'active' classes
function showScreen(id) {
  document
    .querySelectorAll(".screen")
    .forEach((s) => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// -navigates user to the upload screen 
function goToUpload() {
  showScreen("upload");
}

// image upload and object detection
async function submitImages(event) {
  if (event) event.preventDefault();
  showScreen("loading");

  const topFile = document.getElementById("topImage").files[0];
  const wideFile = document.getElementById("wideImage").files[0];

  const formData = new FormData();
  formData.append("top_image", topFile);
  formData.append("wide_image", wideFile);
  // sends selected image to the backend for object detection 
  // receives the bounding box coordinates
  const res = await fetch("http://127.0.0.1:8000/detect-both", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  // storing the detection results 
  topBoxes = data.top.boxes;
  wideBoxes = data.wide.boxes;

  const topImg = document.getElementById("topImg");
  const wideImg = document.getElementById("wideImg");

  let loadedCount = 0;
  // to ensure that all images are fully rendered before drawing the boxes to prevent incorrect scaling issues 
  function checkAllLoaded() {
    loadedCount++;

    if (loadedCount === 2) {
      // show screen first (so layout exists)
      showScreen("select");

      // wait for browser to render layout
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          drawBoxes("topCanvas", topBoxes, selectedTop);
          setupClick("topCanvas", topBoxes, "top");

          drawBoxes("wideCanvas", wideBoxes, selectedWide);
          setupClick("wideCanvas", wideBoxes, "wide");
        }, 100);
      });
    }
  }
  // attaching event listeners to ensure that the images are fully loaded
  topImg.onload = checkAllLoaded;
  wideImg.onload = checkAllLoaded;

  topImg.src = "data:image/jpeg;base64," + data.top.image;
  wideImg.src = "data:image/jpeg;base64," + data.wide.image;
}

// renders deteced and selected bounding boxes on a canvas overlay
function drawBoxes(canvasId, boxes, selectedBox) {
  const canvas = document.getElementById(canvasId);
  const img = canvas.previousElementSibling;
  const ctx = canvas.getContext("2d");
  // computing the scaling factors between original image size and displayed size
  // since detection coordinates are based on original resolution, scaling is required tro correctly map them to the UI
  const scaleX = img.clientWidth / img.naturalWidth;
  const scaleY = img.clientHeight / img.naturalHeight;

  // matching canvas size to displayed image size
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  // prevent overalapping artifacts so previoous drawing is cleared
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // draw detected boxes in green
  boxes.forEach((box) => {
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    // ensure that it is scaled accordingly
    ctx.strokeRect(
      box.xmin * scaleX,
      box.ymin * scaleY,
      (box.xmax - box.xmin) * scaleX,
      (box.ymax - box.ymin) * scaleY,
    );
  });

  // draw selected box (works for both detected + manual)
  if (selectedBox) {
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;

    ctx.fillRect(
      selectedBox.xmin * scaleX,
      selectedBox.ymin * scaleY,
      (selectedBox.xmax - selectedBox.xmin) * scaleX,
      (selectedBox.ymax - selectedBox.ymin) * scaleY,
    );

    ctx.strokeRect(
      selectedBox.xmin * scaleX,
      selectedBox.ymin * scaleY,
      (selectedBox.xmax - selectedBox.xmin) * scaleX,
      (selectedBox.ymax - selectedBox.ymin) * scaleY,
    );
  }
}

// function handles both clicking detected bounding boxes as well as manually drawing the bounding box
function setupClick(canvasId, boxes, type) {
  const canvas = document.getElementById(canvasId);
  const img = canvas.previousElementSibling;
  // computing the inverse scaling factors
  // converts screen coordinates into original image coordinates
  const scaleX = img.naturalWidth / img.clientWidth;
  const scaleY = img.naturalHeight / img.clientHeight;
  // tracks whether user is currently dragging for mnaual box drawing
  let isDragging = false;

  canvas.onmousedown = function (e) {
    if (!manualMode || manualTarget !== type) return;
    // recording the starting point of drag relative to canvas
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;

    isDragging = true;
  };

  canvas.onmouseup = function (e) {
    const rect = canvas.getBoundingClientRect();
    // ending point of the drag
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    if (manualMode && manualTarget === type && isDragging) {
      isDragging = false;
      // converting drag coordinates into bounding box in original image scale
      const xmin = Math.min(startX, endX) * scaleX;
      const ymin = Math.min(startY, endY) * scaleY;
      const xmax = Math.max(startX, endX) * scaleX;
      const ymax = Math.max(startY, endY) * scaleY;
      // create a new bounding box from manual output
      const newBox = { xmin, ymin, xmax, ymax };

      if (type === "wide") {
        selectedWide = newBox;
        drawBoxes("wideCanvas", boxes, selectedWide);
      }

      manualMode = false;
      return;
    }

    // converting click position to original image coordinate space
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // identify all bounding boxes that contain the clicked point
    let candidates = [];

    boxes.forEach((box) => {
      if (x >= box.xmin && x <= box.xmax && y >= box.ymin && y <= box.ymax) {
        candidates.push(box);
      }
    });

    if (candidates.length > 0) {
      // pick smallest box 
      // this ensures that it resolves the issue of 2 bounding box overlap and the bigger one is automatically selected when clicked
      let best = candidates[0];
      let smallestArea = (best.xmax - best.xmin) * (best.ymax - best.ymin);

      candidates.forEach((box) => {
        const area = (box.xmax - box.xmin) * (box.ymax - box.ymin);
        if (area < smallestArea) {
          smallestArea = area;
          best = box;
        }
      });

      if (type === "top") {
        selectedTop = best;
        drawBoxes("topCanvas", boxes, selectedTop);
      } else {
        selectedWide = best;
        drawBoxes("wideCanvas", boxes, selectedWide);
      }
    }
  };
}

// activates the manual draw mode
function enableManualMode(type) {
  manualMode = true;
  manualTarget = type;
  alert("Drag to draw a box");
}

// sends selected objects and location data to the backend for processing
async function processSelection() {
  if (!selectedTop || !selectedWide) {
    alert("Select both boxes");
    return;
  }
  // retrieving the user defined object name 
  const objectName = document.getElementById("objectName").value;
  showScreen("loading");
  // sending relevant data over and data will be converted to JSON format for backend processing
  const res = await fetch("http://127.0.0.1:8000/process-selection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      top_box: selectedTop,
      wide_box: selectedWide,
      wide_boxes: wideBoxes,
      user_object_name: objectName,
    }),
  });

  await res.json();

  // reset application state to prepare for next interaction
  resetState();
  showScreen("home");
}
// clears all previous inputs and stored selections
function resetState() {
  selectedTop = null;
  selectedWide = null;
  wideBoxes = [];
  topBoxes = [];
  document.getElementById("objectName").value = "";
  document.getElementById("topImage").value = "";
  document.getElementById("wideImage").value = "";
}



let recognition;
let isRecording = false;

function setupMic() {
  // using the standard speech recognition for browser compatabilty
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  // checking if the browser supports speech recognition
  if (!SpeechRecognition) {
    alert("Speech recognition not supported");
    return;
  }
  // create recognition instance
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  // making sure it is only a single query per activation
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    document.getElementById("micStatus").innerText = "Listening...";
  };

  recognition.onend = () => {
    document.getElementById("micStatus").innerText = "Stopped";
    isRecording = false;
    document.getElementById("micBtn").innerText = "Start";
  };
  // when input speech is successfully captured,
  recognition.onresult = async (event) => {
    // extract the transcript
    const transcript = event.results[0][0].transcript;

    document.getElementById("micStatus").innerText = "You said: " + transcript;

    // detect simple negative intent
    if (transcript.includes("no") || transcript.includes("not")) {
      handleFallback();
      return;
    }
    // storing the last query for potential fallback use
    lastQuery = transcript;

    // performing semantic serach using the user query
    const res = await fetch(
      `http://127.0.0.1:8000/search?query=${encodeURIComponent(transcript)}`,
    );

    const data = await res.json();
    // converting retreived results into spoken response
    speakResult(data);
  };
}
// handles the starting and stopping of voice recording
function toggleMic() {
  if (!recognition) {
    setupMic();
  }
  // start recording if not currently recording and stop recording if already active
  if (!isRecording) {
    recognition.start();
    isRecording = true;
    document.getElementById("micBtn").innerText = "Stop";
  } else {
    recognition.stop();
  }
}

function speak(text) {
  // creating a speech utterance object 
  const utterance = new SpeechSynthesisUtterance(text);
  // playing audio output
  speechSynthesis.speak(utterance);
}

// generating human readable response from backend results
function speakResult(data) {
  let text = "";
  // if no results found
  if (!data || data.length === 0) {
    text = "I could not find anything";
  } 
  // if single result found
  else if (data.length === 1) {
    text = "I found your object. " + data[0].description;
  } 
  else {
    // multiple results
    text = `I found ${data.length} possible locations. `;
    // mention all options
    data.forEach((item, index) => {
      text += `Option ${index + 1}: ${item.description}. `;
    });
  }
  speak(text);
}
// is triggered when the user say no
async function handleFallback() {
  if (!lastQuery) {
    speak("I don't know what to search for");
    return;
  }
  // alternative retrieval strategy is used
  const res = await fetch(
    `http://127.0.0.1:8000/search-common?query=${encodeURIComponent(lastQuery)}`
  );

  const data = await res.json();

  if (!data || !data.location) {
    speak("I could not find any common location");
  } else {
    speak("It is most likely at the " + data.location);
  }
}