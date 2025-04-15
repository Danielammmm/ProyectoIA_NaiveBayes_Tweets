function enviarTweet() {
    const texto = document.getElementById("tweetInput").value;
  
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: texto })
    })
      .then(response => response.json())
      .then(data => {
        document.getElementById("resultado").innerText =
          `Sentimiento: ${data.sentiment} (Tiempo: ${data.inference_time_ms} ms)`;
      })
      .catch(error => {
        console.error("Error:", error);
        document.getElementById("resultado").innerText =
          "Error al analizar el tweet.";
      });
  }
  