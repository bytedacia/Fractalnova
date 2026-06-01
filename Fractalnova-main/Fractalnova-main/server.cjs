const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 3000;
const WEB_DIR = path.join(__dirname, "web");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
};

const server = http.createServer((req, res) => {
  let url = req.url.split("?")[0];
  if (url === "/") url = "/index.html";

  const filePath = path.join(WEB_DIR, url);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/html; charset=utf-8" });
      res.end("<h1>404 — File non trovato</h1>");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME[ext] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`✦ FractalNova — http://localhost:${PORT}`);
});
