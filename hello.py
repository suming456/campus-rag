from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello from Auto Deploy!")

if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', 8000))
    print("Listening on port 8000...")
    server.serve_forever()