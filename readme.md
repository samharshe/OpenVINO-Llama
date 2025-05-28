to compile inferencer: run make inferencer-build out of backend .
to run server: run cargo run out of backend/server.
this starts the backend on port 3000.
to start the frontend, go to old_frontend and run python http.server 8000 or whatever you want. just make sure it runs out of port 8000.
this all recovers the functionality of the old demo.
everything, including text inference, should work once a proper text model, including .bin and .xml, is loaded into server/fixtures/text_model . for now, I am just CURLing my inference requests to the infer/text endpoint because the frontend is not set up to send text inference requests. that HTML and CSS is written, though, and only needs to be plugged in. (of course this doesn't matter until the text model runs without crashing.)