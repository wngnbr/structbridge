"""API response validation and automatic/manual normal-image selection."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

from normal_refinement.images import ImageRequired, require_generated_image
from normal_refinement.providers.gemini import generate_normal_atlas


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / 'reference.png'
        Image.new('RGB', (64, 64), (128, 128, 255)).save(self.reference)
        self.output = self.root / 'generated_normals.png'
        environment = patch.dict(os.environ, {
            'GEMINI_API_URL': 'https://example.invalid/generateContent',
            'GEMINI_API_KEY': 'test-key',
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.post = patch('normal_refinement.providers.gemini.requests.post').start()
        self.addCleanup(patch.stopall)

    def response(self, size=(64, 64), image_format='PNG'):
        data = io.BytesIO()
        Image.new('RGB', size, (128, 128, 255)).save(data, format=image_format)
        response = Mock()
        response.json.return_value = {'candidates': [{'content': {'parts': [
            {'inlineData': {'mimeType': 'image/' + image_format.lower(),
                            'data': base64.b64encode(data.getvalue()).decode()}},
        ]}}]}
        self.post.return_value = response
        return response

    def test_auto_generates_once_then_reuses_image_without_credentials(self):
        self.response(image_format='JPEG')
        result = require_generated_image(self.reference, self.root, 'chair', 4, image_provider='auto')
        self.assertEqual(result, self.output)
        with Image.open(result) as image:
            self.assertEqual(image.format, 'PNG')
        request = json.loads((self.root / 'image_request.json').read_text())
        self.assertEqual(request['status'], 'image_available')
        self.assertEqual(request['provider'], 'gemini')
        with patch.dict(os.environ, {}, clear=True):
            require_generated_image(self.reference, self.root, 'chair', 4, image_provider='auto')
        self.post.assert_called_once()
        self.assertFalse(self.output.with_name(self.output.name + '.tmp').exists())

    def test_six_view_payload_and_output(self):
        self.response(size=(96, 64))
        generate_normal_atlas(self.reference, self.output, 'table', 6)
        kwargs = self.post.call_args.kwargs
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')
        self.assertEqual(kwargs['json']['generationConfig']['imageConfig']['aspectRatio'], '3:2')
        parts = kwargs['json']['contents'][0]['parts']
        self.assertIn('6 camera-space normal maps', parts[0]['text'])
        self.assertEqual(base64.b64decode(parts[1]['inlineData']['data']), self.reference.read_bytes())

    def test_explicit_image_and_missing_explicit_path_never_call_api(self):
        result = require_generated_image(self.reference, self.root, 'chair', 4,
                                         generated=self.reference, image_provider='gemini')
        self.assertEqual(result, self.reference)
        with self.assertRaises(ImageRequired):
            require_generated_image(self.reference, self.root, 'chair', 4,
                                    generated=self.root / 'missing.png', image_provider='auto')
        self.post.assert_not_called()

    def test_manual_and_unconfigured_auto_preserve_file_handoff(self):
        with self.assertRaises(ImageRequired):
            require_generated_image(self.reference, self.root, 'chair', 4, image_provider='manual')
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ImageRequired):
            require_generated_image(self.reference, self.root, 'chair', 4, image_provider='auto')
        self.post.assert_not_called()

    def test_missing_credentials_fail_before_network_request(self):
        for provider, environment in [('gemini', {}), ('auto', {'GEMINI_API_URL': 'https://example.invalid'})]:
            with self.subTest(provider=provider), patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'GEMINI_API_'):
                    require_generated_image(self.reference, self.root, 'chair', 4, image_provider=provider)
        self.post.assert_not_called()

    def test_bad_api_responses_do_not_publish_an_image(self):
        bodies = [
            {}, {'error': {'message': 'secret server details'}},
            {'candidates': [{'content': {'parts': [{'text': 'no image'}]}}]},
            {'candidates': [{'content': {'parts': [{'inlineData': {'mimeType': 'image/png', 'data': 'broken'}}]}}]},
            {'candidates': [{'content': {'parts': [{'inlineData': {
                'mimeType': 'image/png', 'data': base64.b64encode(b'not an image').decode(),
            }}]}}]},
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.response().json.return_value = body
                with self.assertRaisesRegex(RuntimeError, 'no valid image'):
                    require_generated_image(self.reference, self.root, 'chair', 4, image_provider='auto')
                self.assertFalse(self.output.exists())
                self.assertEqual(json.loads((self.root / 'image_request.json').read_text())['status'], 'generation_failed')

    def test_wrong_aspect_ratio_is_rejected_before_caching(self):
        self.response(size=(96, 64))
        with self.assertRaisesRegex(RuntimeError, 'aspect ratio'):
            generate_normal_atlas(self.reference, self.output, num_views=4)
        self.assertFalse(self.output.exists())

    def test_http_errors_and_timeouts_are_not_silently_retried(self):
        response = requests.Response()
        response.status_code = 401
        errors = [(requests.HTTPError('secret endpoint', response=response), 'HTTP 401'),
                  (requests.Timeout('secret endpoint'), 'timed out'),
                  (requests.ConnectionError('secret endpoint'), 'Cannot reach')]
        for error, message in errors:
            with self.subTest(error=type(error).__name__):
                self.post.reset_mock()
                self.post.side_effect = error
                with self.assertRaisesRegex(RuntimeError, message) as raised:
                    generate_normal_atlas(self.reference, self.output)
                self.assertNotIn('secret endpoint', str(raised.exception))
                self.post.assert_called_once()
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
