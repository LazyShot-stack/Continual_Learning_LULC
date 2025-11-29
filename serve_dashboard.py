import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    # Allow port to be specified
    port = PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
        
    print(f"Starting dashboard server at http://localhost:{port}")
    print(f"Root directory: {os.path.abspath(DIRECTORY)}")
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/dashboard/index.html"
            print(f"Opening {url} in browser...")
            webbrowser.open(url)
            print("Press Ctrl+C to stop the server.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error: {e}")
        print(f"Try using a different port: python serve_dashboard.py 8080")

if __name__ == "__main__":
    main()
