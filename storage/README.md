# storage/

Folder ini menyimpan `encodings.json` saat runtime.

⚠️  PENTING:
- File ini di-generate otomatis saat service pertama kali jalan
- Jangan commit `encodings.json` ke git (sudah ada di .gitignore)
- Di Render.com free tier, storage ini EPHEMERAL (hilang saat redeploy)
  → Plan migrasi ke VPS atau gunakan Render.com Disk ($1/bln) untuk persistensi

.gitkeep ini ada supaya folder tidak kosong di git.
