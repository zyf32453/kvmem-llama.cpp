"""Exercise the acceptance client's actual chunked HTTP error framing."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ninfer_lifecycle_service_acceptance import Stream, check_error_stream


@contextmanager
def server(frames=(), status=200, truncate=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(status)
            if status != 200:
                data = json.dumps({'error': {'code': 'service_unavailable'}}).encode()
                self.send_header('Content-Length', str(len(data)))
                self.end_headers();self.wfile.write(data)
            else:
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                for frame in frames:
                    value = frame if isinstance(frame, str) else json.dumps(frame)
                    data = ('data: '+value+'\n\n').encode()
                    self.wfile.write(f'{len(data):x}\r\n'.encode()+data+b'\r\n')
                if truncate:
                    self.wfile.write(b'20\r\nincomplete')
                else:
                    self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            self.close_connection = True

    instance = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance.server_address[1]
    finally:
        instance.shutdown();instance.server_close();thread.join(5)


class ServiceWireTest(unittest.TestCase):
    error = {'error': {'code': 'service_unavailable', 'message': 'quota exceeded'}}

    def read(self, frames=(), status=200, truncate=False):
        with server(frames, status, truncate) as port:
            return Stream(port, 'test').start().join(5)

    def test_error_is_terminal_without_success_done_sentinel(self):
        response = self.read([{'choices': [{'delta': {'content': 'partial'}}]}, self.error])
        self.assertFalse(response['done'])
        self.assertTrue(response['finished'])
        check_error_stream(response)

    def test_pre_header_unavailability(self):
        check_error_stream(self.read(status=503))

    def test_truncated_http_body_is_not_a_valid_error(self):
        response = self.read([self.error], truncate=True)
        self.assertIn('transport_error', response)
        with self.assertRaises(RuntimeError):check_error_stream(response)

    def test_success_is_not_a_failed_request(self):
        response = self.read([{'choices': [{'delta': {'content': '599'}}]}, '[DONE]'])
        self.assertTrue(response['done'])
        with self.assertRaises(RuntimeError):check_error_stream(response)

    def test_error_followed_by_content_is_not_terminal(self):
        response = self.read([self.error, {'choices': [{'delta': {'content': 'later'}}]}])
        with self.assertRaises(RuntimeError):check_error_stream(response)

    def test_different_error_code_is_not_unavailability(self):
        response = self.read([{'error': {'code': 'server_overloaded'}}])
        with self.assertRaises(RuntimeError):check_error_stream(response)


if __name__ == '__main__':
    unittest.main()
