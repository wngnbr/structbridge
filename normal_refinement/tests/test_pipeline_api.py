"""Exercise the public shell entry point with a local image API test double."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import torch
import trimesh


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA renderer required')
class PipelineApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.mesh = self.root / 'coarse.obj'
        trimesh.creation.icosphere(subdivisions=1).export(self.mesh)
        self.calls = []
        calls = self.calls

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(payload)
                if self.path == '/fail':
                    self.send_response(503)
                    self.end_headers()
                    return
                # Identity normals are a protocol test fixture, not generated
                # details or a fallback that the production pipeline can use.
                inline = payload['contents'][0]['parts'][1]['inlineData']
                body = json.dumps({'candidates': [{'content': {'parts': [{'inlineData': inline}]}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.env = dict(os.environ)
        for key in ('SHARP_PYTHON', 'REFINE_PYTHON'):
            self.env.pop(key, None)
        self.env.update(
            PATH=str(Path(sys.executable).parent) + os.pathsep + self.env.get('PATH', ''),
            GEMINI_API_URL=f'http://127.0.0.1:{self.server.server_port}/generate',
            GEMINI_API_KEY='local-test-only',
            NO_PROXY='127.0.0.1,localhost',
            no_proxy='127.0.0.1,localhost',
        )

    def invoke(self, *extra):
        command = ['bash', 'scripts/sample.sh', '--stage1_mesh', str(self.mesh),
                   '--cat', 'object', '--output_dir', str(self.root / 'output'),
                   '--refine_steps', '2', '--refine_resolution', '64', *extra]
        return subprocess.run(command, cwd=ROOT, env=self.env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)

    def test_shell_generates_refines_exports_and_reuses_cache(self):
        result = self.invoke('--num_views', '6')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]['generationConfig']['imageConfig']['aspectRatio'], '3:2')
        final = self.root / 'output' / 'final.obj'
        self.assertTrue(final.is_file())
        self.assertGreater(len(trimesh.load(final, force='mesh').faces), 0)
        state = json.loads((final.parent / 'pipeline.json').read_text())
        self.assertEqual(state['status'], 'complete')
        before = final.read_bytes()
        result = self.invoke('--num_views', '6')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(self.calls), 1, 'A cached run must not request another image')
        self.assertEqual(before, final.read_bytes())

    def test_http_failure_stops_shell_without_publishing_final_mesh(self):
        self.env['GEMINI_API_URL'] = f'http://127.0.0.1:{self.server.server_port}/fail'
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('HTTP 503', result.stdout)
        self.assertEqual(len(self.calls), 1)
        output = self.root / 'output'
        self.assertFalse((output / 'final.obj').exists())
        self.assertFalse((output / 'stage2' / 'generated_normals.png').exists())
        self.assertEqual(json.loads((output / 'pipeline.json').read_text())['status'], 'stage2_failed')


if __name__ == '__main__':
    unittest.main()
