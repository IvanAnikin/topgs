

window.onload = () => {
  
  var video = document.querySelector("#videoElement");
  
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (stream) {
        video.srcObject = stream;
      })
      .catch(function (err0r) {
        console.log("Something went wrong!");
      });
  }
  function sendImage() {
    const canvas = document.createElement('canvas');
    const video = document.getElementById('videoElement');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
  
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/detect', true);
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
    xhr.onload = function() {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        const color = response.color_name;
        const coords = response.coordinates;
        const hex = response.hex_code;
  
        console.log(response)

        const box = document.createElement('div');
        box.style.position = 'absolute';
        box.style.left = `${coords[0][0]}px`;
        box.style.top = `${coords[0][1]}px`;
        box.style.width = `${coords[1][0] - coords[0][0]}px`;
        box.style.height = `${coords[1][1] - coords[0][1]}px`;
        box.style.border = `3px solid ${color}`;
        box.style.color = `${color}`;
        box.innerHTML = `<span>${color}</span>`;
        document.body.appendChild(box);
        
      } else {
        console.error(xhr.statusText);
      }
    };
    xhr.onerror = function() {
      console.error(xhr.statusText);
    };
    xhr.send(JSON.stringify({image: dataURL}));
  }
  var detectButton = document.getElementById('sendbutton');
  detectButton.addEventListener('click', sendImage);

};