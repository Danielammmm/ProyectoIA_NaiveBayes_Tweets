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
      const resultadoDiv = document.getElementById("resultado");

      // Resultado principal
      resultadoDiv.innerHTML = `Sentimiento: ${data.sentiment} (Tiempo: ${data.inference_time_ms} ms)`;

      // Tabla de probabilidades
      const table = document.createElement("table");
      table.style.marginTop = "10px";
      table.style.borderCollapse = "collapse";
      table.style.width = "100%";
      table.innerHTML = `
        <tr><th style="text-align:left;">Clase</th><th style="text-align:left;">Probabilidad</th></tr>
        ${Object.entries(data.probabilities)
          .map(([clase, prob]) => `<tr><td>${clase}</td><td>${(prob * 100).toFixed(2)}%</td></tr>`)
          .join("")}
      `;

      resultadoDiv.appendChild(table);
    })
    .catch(error => {
      console.error("Error:", error);
      document.getElementById("resultado").innerText =
        "Error al analizar el tweet.";
    });
}
