# Inference Demo Device for AO

### note from Sam:
to avoid inducing CORS errors, don't open the demo in `webapp` directly via `file://`. instead, serve it via a basic local HTTP server. the permissions hard-coded in `server/src/main.rs` require port 8000.

the easy way I am doing this:
```bash
cd webapp && python -m http.server 8000
```