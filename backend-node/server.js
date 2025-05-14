const express = require('express');
const cors = require('cors');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { exec } = require('child_process');
const multer = require('multer');

const app = express();
app.use(cors());

const resultDir = path.resolve('results');
if (!fs.existsSync(resultDir)) fs.mkdirSync(resultDir);


const upload = multer({ storage: multer.memoryStorage() });

app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'Nenhuma imagem enviada' });
    }

    // Salvar imagem temporária só para o script
    const tempInputPath = path.join(os.tmpdir(), Date.now() + path.extname(req.file.originalname));
    fs.writeFileSync(tempInputPath, req.file.buffer);

    // Executar script com o arquivo temporário
    const command = `python yolo_detect.py --source "${tempInputPath}" --output "${resultDir}"`;

    exec(command, (error, stdout, stderr) => {
        // Apagar o arquivo temporário após uso
        fs.unlink(tempInputPath, () => { });

        if (error) {
            console.error("Erro:", stderr);
            return res.status(500).json({ error: 'Erro no processamento' });
        }

        const [imagePath, similarity] = stdout.trim().split('|');

        res.json({
            success: true,
            image: `/processed/${path.basename(imagePath)}`,
            similarity: parseFloat(similarity),
        });
    });
});

app.use('/processed', express.static(resultDir));

app.listen(5000, () => console.log('🚀 Servidor rodando na porta 5000'));
