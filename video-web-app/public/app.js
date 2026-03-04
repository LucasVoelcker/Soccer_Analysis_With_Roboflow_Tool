const form = document.getElementById("upload-form");
const input = document.getElementById("video-input");
const statusText = document.getElementById("status");
const resultVideo = document.getElementById("result-video");
const processButton = document.getElementById("process-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!input.files || input.files.length === 0) {
    statusText.textContent = "Select a video first.";
    return;
  }

  const file = input.files[0];
  const data = new FormData();
  data.append("video", file);

  statusText.textContent = "processing video...";
  processButton.disabled = true;
  resultVideo.style.display = "none";
  resultVideo.removeAttribute("src");
  resultVideo.load();

  try {
    const response = await fetch("/process-video", {
      method: "POST",
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.details || payload.error || "Unexpected server error.");
    }

    resultVideo.src = `${payload.videoUrl}?t=${Date.now()}`;
    resultVideo.style.display = "block";
    statusText.textContent = "Video processed.";
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
  } finally {
    processButton.disabled = false;
  }
});
