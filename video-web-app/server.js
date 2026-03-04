const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3000;

const appRoot = __dirname;
const uploadsDir = path.join(appRoot, "uploads");
const outputsDir = path.join(appRoot, "outputs");
const pythonScriptPath = path.join(appRoot, "..", "generate-output-video.py");

fs.mkdirSync(uploadsDir, { recursive: true });
fs.mkdirSync(outputsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || ".mp4";
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `input-${unique}${ext}`);
  },
});

const upload = multer({ storage });

app.use(express.static(path.join(appRoot, "public")));
app.use("/videos", express.static(outputsDir));

app.post("/process-video", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Nenhum video enviado." });
  }

  const inputVideoPath = req.file.path;
  const outputName = `output-${Date.now()}-${Math.round(Math.random() * 1e9)}.mp4`;
  const outputVideoPath = path.join(outputsDir, outputName);
  const pythonCmd = process.env.PYTHON_CMD || "python";

  const args = [
    pythonScriptPath,
    "--video-path",
    inputVideoPath,
    "--output-video-path",
    outputVideoPath,
  ];

  let stderr = "";
  let responseSent = false;
  const py = spawn(pythonCmd, args, { cwd: path.dirname(pythonScriptPath) });

  py.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  py.on("error", (err) => {
    responseSent = true;
    return res.status(500).json({
      error: "Nao foi possivel iniciar o processo Python.",
      details: err.message,
    });
  });

  py.on("close", async (code) => {
    if (responseSent) {
      return;
    }
    responseSent = true;

    try {
      await fs.promises.unlink(inputVideoPath);
    } catch (_err) {
      // Best effort cleanup
    }

    if (code !== 0) {
      return res.status(500).json({
        error: "Falha ao processar video.",
        details: stderr || `Process exited with code ${code}.`,
      });
    }

    if (!fs.existsSync(outputVideoPath)) {
      return res.status(500).json({
        error: "Processamento finalizado, mas video de saida nao foi encontrado.",
      });
    }

    return res.json({
      message: "Video processado com sucesso.",
      videoUrl: `/videos/${outputName}`,
    });
  });
});

app.listen(PORT, () => {
  console.log(`Servidor iniciado em http://localhost:${PORT}`);
});
